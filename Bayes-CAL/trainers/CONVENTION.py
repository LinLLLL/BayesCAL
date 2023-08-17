import os.path as osp
import numpy as np
from scipy.stats import ortho_group
from sympy import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, sigma=1e-2):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BCAL.N_CTX
        ctx_init = cfg.TRAINER.BCAL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.BCAL.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)

            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            #            ctx_vectors += 10
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # Mean field assumption for variational inference
        # Record initialized information
        self.ctx_init = ctx_vectors
        self.ctx_sigma_init = sigma * torch.ones_like(ctx_vectors)

        # Mean vectors
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # Sigma vectors
        self.ctx_sigma = nn.Parameter(sigma * torch.ones_like(ctx_vectors))  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        print("========================Prompts with out learnable parameters=====================")
        print(prompts)

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BCAL.CLASS_TOKEN_POSITION

    def kldivergence(self):
        # return computed kl divergence
        scale = 1e-8
        s1 = self.ctx_sigma_init.float().cuda()
        s2 = self.ctx_sigma.float().cuda()

        s1 = s1 ** 2
        s2 = s2 ** 2
        #        print(self.ctx_sigma)

        mu1 = self.ctx_init.float().cuda()
        mu2 = self.ctx.float().cuda()

        kl = torch.log(s1 / s2 + 1e-3) + (s2 ** 2 + (mu1 - mu2) ** 2) / (2 * s1 ** 2 + 1e-3)

        assert kl.shape == s1.shape
        kl = torch.sum(kl) * scale
        return torch.clamp(kl, min=0.0)

    def forward(self):
        ctx_mean = self.ctx
        ctx_sigma = self.ctx_sigma
        # Sample from the variational distribution
        ctx = ctx_mean + ctx_sigma ** 2 * torch.randn_like(self.ctx)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        return prompts


class Classifier(nn.Module):
    def __init__(self, nfeat, classnames):
        super().__init__()
        n_cls = len(classnames)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layer1 = nn.Sequential(
            nn.Linear(nfeat, int(nfeat / 4)),
            nn.BatchNorm1d(int(nfeat / 4)))
        self.layer2 = nn.Sequential(
            nn.Linear(int(nfeat / 4), int(nfeat / 16)),
            nn.BatchNorm1d(int(nfeat / 16)))
        self.layer3 = nn.Sequential(
            nn.Linear(int(nfeat / 16), n_cls),
            nn.BatchNorm1d(n_cls))

    def forward(self, x):
        for layer in [self.layer1, self.layer2, self.layer3]:
            x = self.relu(layer(x)) if layer == self.layer3 else self.sigmoid(layer(x))

        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, domainnames, clip_model, nfeat, sigma=1e-5):
        super().__init__()

        self.image_encoder_backbone = clip_model.visual
        self.classifier = Classifier(nfeat, classnames)

        self.dtype = clip_model.dtype

        # Mean field assumption for variational inference
        self.logit_sigma = nn.Parameter(sigma * torch.ones(1, len(classnames)))  # to be optimized

    def forward(self, image):
        image_features = self.image_encoder_backbone(image.type(self.dtype))
        image_features = image_features.float()
        image_features.requires_grad_(requires_grad=True)
        self.logit = self.classifier(image_features)  # to be optimized

        logit_mean = self.logit
        logit_sigma = self.logit_sigma.repeat(logit_mean.shape[0], 1)
        # Sample from the variational distribution
        output = (logit_mean + logit_sigma ** 2 * torch.randn_like(self.logit))

        return output

    def init_from_flat_params(self, init_params):
        cnt = 0
        for name, parameters in self.classifier.named_parameters():
            num = 1
            for i in range(len(list(parameters.shape))):
                num *= parameters.shape[i]
            parameters.data = init_params[cnt:cnt + num].reshape(list(parameters.shape)).type(
                parameters.dtype).to(device)
            cnt += num


@TRAINER_REGISTRY.register()
class CONVENTION(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.BCAL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        domainnames = self.dm.dataset.alldomain

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BCAL.PREC == "fp32" or cfg.TRAINER.BCAL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        nfeat = 1024
        self.model = CustomCLIP(cfg, classnames, domainnames, clip_model, nfeat)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.classifier, cfg.OPTIM)  ##############
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("classifier", self.model.classifier, self.optim, self.sched)  ###################
        self.scaler = GradScaler() if cfg.TRAINER.BCAL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, domain_label, _, _ = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.BCAL.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits_category = self.model(image)

            l1 = F.cross_entropy(logits_category, label)
            if self.model.training:
                l1.requires_grad_(requires_grad=True)
                loss = l1

                # log task-specific parameters
                self.optim_path_params = np.array([])
                for name, parameters in self.model.named_parameters():
                    if "classifier" in name:
                        param = parameters.detach().cpu().numpy()
                        self.optim_path_params = np.concatenate((self.optim_path_params, param.flatten()), 0)
                self.optim_path_params = self.optim_path_params.flatten()
                self.model_backward_and_update(loss)

            else:
                loss = l1

        if self.model.training:
            loss_summary = {
                "loss": loss.item(),
                "loss_CE": l1.item(),
                "acc": compute_accuracy(logits_category, label)[0].item(),
                "optim_path_params": self.optim_path_params
            }
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            loss_summary = {
                "loss": loss.item(),
                "loss_CE": l1.item(),
                "acc": compute_accuracy(logits_category, label)[0].item(),
            }

        return loss_summary

    def IRMLoss(self, logits, label, domain):
        # Invariant Risk Minimization
        def penalty(logits, y):
            scale = torch.tensor(1.).to(self.device).requires_grad_()
            loss = F.cross_entropy(logits * scale, y)
            grad = autograd.grad(loss, [scale], create_graph=True)[0]
            return torch.sum(grad ** 2)

        Logits, Label = [], []
        for i in torch.unique(domain).cpu().numpy().tolist():
            Logits.append(logits[domain == i])
            Label.append(label[domain == i])
        reguloss = sum([penalty(Logits[i], Label[i]) for i in range(len(Label))])
        return reguloss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        classname = batch["classname"]
        domainname = batch["domainname"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain, classname, domainname

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

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

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import gensim

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertModel, BertTokenizer

_tokenizer = _Tokenizer()
from transformers import logging

device = "cuda" if torch.cuda.is_available() else "cpu"
import warnings

warnings.filterwarnings("ignore")


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


class Projection(nn.Module):
    def __init__(self, nfeat=300):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.layer1 = nn.Sequential(
            nn.Linear(nfeat, 600),
            nn.BatchNorm1d(600))
        self.layer2 = nn.Sequential(
            nn.Linear(600, 1024),
            nn.BatchNorm1d(1024))

    def forward(self, x):
        for layer in [self.layer1, self.layer2]:
            x = self.relu(layer(x)) if layer == self.layer2 else self.sigmoid(layer(x))
        return x


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
    def __init__(self, cfg, classnames, sigma=1e-5):
        super().__init__()
        self.classnames = classnames
        self.sigma = sigma
        n_cls = len(classnames)
        self.n_cls = n_cls
        # use given words to initialize context vectors
        classnames = [name.replace("_", " ") for name in classnames]

        self.projection = Projection().to(device)
        self.pretrain_model_path = "/DATA/NICO_SUB.pth" if n_cls==19 else "/DATA/NICO_domain.pth"
        # w2v = gensim.models.KeyedVectors.load_word2vec_format(pretrain_model_path, binary=False)
        # embeddings = [w2v[name] for name in self.classnames]
        self.embeddings = torch.load(self.pretrain_model_path).to(device)
        self.ctx_vectors = torch.empty(n_cls, 1024)
        nn.init.normal_(self.ctx_vectors, std=0.02)

        # Mean field assumption for variational inference
        # Record initialized information
        self.ctx_init = self.ctx_vectors.detach()
        self.ctx_sigma_init = sigma * torch.ones_like(self.ctx_init)

        # Mean vectors
        self.ctx = self.ctx_vectors  # to be optimized
        # Sigma vectors
        self.ctx_sigma = nn.Parameter(sigma * torch.ones_like(self.ctx_init))  # to be optimized

    def kldivergence(self):
        # return computed kl divergence
        scale = 1e-8
        s1 = self.ctx_sigma_init.float().to(device)
        s2 = self.ctx_sigma.float().to(device)

        s1 = s1 ** 2
        s2 = s2 ** 2

        mu1 = self.ctx_init.float().to(device)
        mu2 = self.ctx.float().to(device)

        kl = torch.log(s1 / s2 + 1e-3) + (s2 ** 2 + (mu1 - mu2) ** 2) / (2 * s1 ** 2 + 1e-3)

        assert kl.shape == s1.shape
        kl = torch.sum(kl) * scale
        return torch.clamp(kl, min=0.0)

    def forward(self):
        self.ctx = self.projection(self.embeddings)  # to be optimized
        ctx_mean = self.ctx
        ctx_sigma = self.ctx_sigma

        # Sample from the variational distribution
        ctx = (ctx_mean + ctx_sigma ** 2 * torch.randn_like(self.ctx))
        return ctx


class Extractor(nn.Module):
    def __init__(self, nfeat):
        super().__init__()

    def forward(self, x):
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, domainnames, clip_model, nfeat):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames)
        self.prompt_learner_context = PromptLearner(cfg, domainnames)
        self.image_encoder_backbone = clip_model.visual
        self.image_encoder_context_branch = Extractor(nfeat)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def update_init(self, image, labels_t, domain_labels_t, pre_train_info):
        image_features = self.image_encoder_backbone(image.type(self.dtype))
        image_features = image_features.float()

    def forward(self, image):
        image_features = self.image_encoder_backbone(image.type(self.dtype))
        # image_features = (self.rotation(image_features)).float()
        image_features = image_features.float()

        prompts = self.prompt_learner()
        prompts_context = self.prompt_learner_context()

        text_features_category = prompts.type(self.dtype)
        text_features_context = prompts_context.type(self.dtype)
        image_features_category = (image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)).type(self.dtype)
        image_features_context = (image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)).type(self.dtype)
        text_features_category = text_features_category / (text_features_category.norm(dim=-1, keepdim=True) + 1e-6)
        text_features_context = text_features_context / (text_features_context.norm(dim=-1, keepdim=True) + 1e-6)
        logit_scale = self.logit_scale.exp()
        logits_category = logit_scale * image_features_category @ text_features_category.t()
        logits_context = logit_scale * image_features_context @ text_features_context.t()

        image_features.requires_grad_(requires_grad=True)

        output = [logits_category, logits_context, text_features_category, text_features_context, image_features]

        return output

    def init_from_flat_params(self, init_params):
        cnt = 0
        for name, parameters in self.prompt_learner.named_parameters():
            num = 1
            for i in range(len(list(parameters.shape))):
                num *= parameters.shape[i]
            parameters.data = init_params[cnt:cnt + num].reshape(list(parameters.shape)).cuda().type(
                parameters.dtype) if device == "cuda" else init_params[cnt:cnt + num].reshape(
                list(parameters.shape)).type(parameters.dtype)
            cnt += num

        cnt = 0
        for name, parameters in self.prompt_learner_context.named_parameters():
            num = 1
            for i in range(len(list(parameters.shape))):
                num *= parameters.shape[i]
            parameters.data = init_params[cnt:cnt + num].reshape(list(parameters.shape)).cuda().type(
                parameters.dtype) if device == "cuda" else init_params[cnt:cnt + num].reshape(
                list(parameters.shape)).type(parameters.dtype)
            cnt += num


class Logits(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.logit_scale = clip_model.logit_scale

    def forward(self, image_features, text_features_category, text_features_context):
        logit_scale = self.logit_scale.exp()
        logits_category = logit_scale * (image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)).type(
            text_features_category.dtype) @ text_features_category.t()
        logits_context = logit_scale * (image_features / (1e-6 + image_features.norm(dim=-1, keepdim=True))).type(
            text_features_context.dtype) @ text_features_context.t()
        logits_category.requires_grad_(requires_grad=True)
        logits_context.requires_grad_(requires_grad=True)
        return logits_category, logits_context


@TRAINER_REGISTRY.register()
class BCAL_W2V(TrainerX):
    """Context Optimization (BCAL_LV).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.BCAL_LV.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        domainnames = self.dm.dataset.alldomain

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BCAL_LV.PREC == "fp32" or cfg.TRAINER.BCAL_LV.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        nfeat = 1024
        self.model = CustomCLIP(cfg, classnames, domainnames, clip_model, nfeat)
        self.model.to(self.device)

        self.logits = Logits(clip_model)
        self.model.to(self.device)
        self.logits.to(self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.model.prompt_learner_context, cfg.MODEL.INIT_WEIGHTS)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)  ##############
        self.optim_context = build_optimizer(self.model.prompt_learner_context, cfg.OPTIM)  ##############
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.sched_context = build_lr_scheduler(self.optim_context, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)  ###################
        self.register_model("prompt_learner_context", self.model.prompt_learner_context, self.optim_context,
                            self.sched_context)  ###################

        self.scaler = GradScaler() if cfg.TRAINER.BCAL_LV.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.logits = nn.DataParallel(self.logits)
        self.data = {}

    def forward_backward(self, batch):
        # image, label, domain_label = self.parse_batch_train(batch)
        image, label, domain_label, classname, domainname = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.BCAL_LV.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits_category, logits_context, text_features_category, text_features_context, image_features = self.model(image)
            if self.model.training:
                logits_category, logits_context = self.logits(image_features, text_features_category, text_features_context)

            l1 = F.cross_entropy(logits_category, label)
            if self.model.training:
                l2 = F.cross_entropy(logits_context, domain_label)
                # gradorth
                l1_grad = torch.autograd.grad(l1, image_features, retain_graph=True)
                l2_grad = torch.autograd.grad(l2, image_features, retain_graph=True)
                grad_category = l1_grad[0] / (l1_grad[0].norm(dim=-1, keepdim=True) + 1e-6)
                grad_context = l2_grad[0] / (l2_grad[0].norm(dim=-1, keepdim=True) + 1e-6)
                l_orth = (((grad_category @ grad_context.t()).diag()) ** 2).sum()
                l_IRM = self.IRMLoss(logits_category, label, domain_label)
                l_KL = self.model.prompt_learner.kldivergence() + self.model.prompt_learner_context.kldivergence()
                l_orth.requires_grad_(requires_grad=True)
                l_KL.requires_grad_(requires_grad=True)
                l_IRM.requires_grad_(requires_grad=True)
                l1.requires_grad_(requires_grad=True)
                l2.requires_grad_(requires_grad=True)
                loss = l1 + self.cfg.alpha1 * l2 + self.cfg.alpha2 * l_IRM \
                       + self.cfg.alpha3 * l_orth + l_KL

                # log task-specific parameters and image_features
                for classname0 in classname:
                    for domainname0 in domainname:
                        if not (classname0, domainname0) in self.data.keys():
                            self.data[(classname0, domainname0)] = []
                    for i in range(image_features.shape[0]):
                        if classname[i] == classname0 and domainname[i] == domainname0:
                            self.data[(classname0, domainname0)].append(
                                image_features[i].detach().cpu().unsqueeze(1).numpy())
                self.optim_path_features = self.data

                self.optim_path_params1 = np.array([])
                self.optim_path_params2 = np.array([])
                for name, parameters in self.model.named_parameters():
                    if "prompt_learner" in name and "context" not in name:
                        param = parameters.detach().cpu().numpy()
                        self.optim_path_params1 = np.concatenate((self.optim_path_params1, param.flatten()), 0)
                    if "prompt_learner_context" in name:
                        param = parameters.detach().cpu().numpy()
                        self.optim_path_params2 = np.concatenate((self.optim_path_params2, param.flatten()), 0)
                self.optim_path_params = np.concatenate([self.optim_path_params1, self.optim_path_params2], 0).flatten()
                self.model_backward_and_update(loss)

            else:
                loss = l1
        if self.model.training:
            loss_summary = {
                "loss": loss.item(),
                "loss_CE": l1.item(),
                "acc": compute_accuracy(logits_category, label)[0].item(),
                "domain_acc": compute_accuracy(logits_context, domain_label)[0].item(),
                "optim_path_features": self.optim_path_features,
                "optim_path_params": self.optim_path_params,
                "text_features_category": text_features_category,
            }
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            loss_summary = {
                "loss": loss.item(),
                "loss_CE": l1.item(),
                "acc": compute_accuracy(logits_category, label)[0].item(),
                "domain_acc": compute_accuracy(logits_context, domain_label)[0].item(),
                "text_features_category": text_features_category,
                "image_features": image_features,
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



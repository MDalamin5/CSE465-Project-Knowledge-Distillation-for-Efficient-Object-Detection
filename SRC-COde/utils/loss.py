# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import torch.nn.functional as F


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def imitation_loss(teacher, student, mask):
    if student is None or teacher is None:
        return 0
    # print(teacher.shape, student.shape, mask.shape)
    diff = torch.pow(student - teacher, 2) * mask
    diff = diff.sum() / mask.sum() / 2

    return diff

#----------------------------------------------------------

# def imitation_loss_kl(teacher, student, mask, temperature=4):
#     """
#     KL Divergence-based imitation loss for knowledge distillation.
#     Args:
#         teacher: Teacher model's output (logits or features).
#         student: Student model's output (logits or features).
#         mask: Mask to focus on specific regions (e.g., object areas).
#         temperature: Temperature for softening distributions.
#     Returns:
#         KL divergence loss value.
#     """
#     if student is None or teacher is None:
#         return 0
    
#     # Compute softmax over teacher and student logits/features
#     teacher_soft = F.softmax(teacher / temperature, dim=-1)
#     student_soft = F.log_softmax(student / temperature, dim=-1)
    
#     # Compute per-pixel KL divergence
#     kl_div = F.kl_div(student_soft, teacher_soft, reduction='none')  # Element-wise KL
    
#     # Apply mask and normalize by active regions
#     loss = (kl_div * mask).sum() / mask.sum()
#     return loss



def combined_imitation_loss(teacher, student, mask, temperature=3, alpha=0.5):
    """
    Combined KL divergence and MSE loss for imitation.
    Args:
        teacher: Teacher model's output (logits or features).
        student: Student model's output (logits or features).
        mask: Mask to focus on specific regions.
        temperature: Temperature for KL divergence.
        alpha: Weight for KL divergence in the combined loss.
    Returns:
        Combined loss value.
    """
    if student is None or teacher is None:
        return 0
    # KL Divergence
    teacher_soft = F.softmax(teacher / temperature, dim=-1)
    student_soft = F.log_softmax(student / temperature, dim=-1)
    kl_div = F.kl_div(student_soft, teacher_soft, reduction='none')
    kl_loss = (kl_div * mask).sum() / mask.sum()
    
    # MSE Loss
    mse_loss = torch.pow(teacher - student, 2) * mask
    mse_loss = mse_loss.sum() / mask.sum()
    
    # Combine losses+
    combined_loss = alpha * kl_loss + (1 - alpha) * mse_loss
    return combined_loss

## ----------------------------------------------------------------------
def imitation_loss_smooth_l1(teacher, student, mask, beta=1.1):# try l2 loss elistics loss
    # Debugging: Check inputs
    if student is None or teacher is None:
        return 0

    diff = student - teacher
    abs_diff = torch.abs(diff)
    smooth_l1_loss = torch.where(
        abs_diff < beta,
        0.5 * (diff**2) / beta,
        abs_diff - 0.5 * beta
    )
    loss = (smooth_l1_loss * mask).sum() / mask.sum()
    return loss

#---------------------------------------------------
# def cosine_similarity_loss(teacher, student, mask):
#     """
#     Cosine similarity loss for knowledge distillation.
#     Args:
#         teacher: Teacher model's output (features or logits).
#         student: Student model's output (features or logits).
#         mask: Mask to focus on specific regions.
#     Returns:
#         Cosine similarity loss value.
#     """
#     if student is None or teacher is None:
#         return 0

#     # Flatten spatial dimensions and permute to [B, H*W, C]
#     teacher = teacher.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
#     student = student.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

#     # Compute cosine similarity across feature dimension
#     cos_sim = F.cosine_similarity(student, teacher, dim=2)  # [B, H*W]

#     # Align mask dimensions with spatial dimensions of cos_sim
#     if mask.ndim == 4:  # [B, 1, H, W]
#         mask = mask.flatten(2).squeeze(1)  # Flatten to [B, H*W]
#     elif mask.ndim == 2:  # [B, 1]
#         mask = mask.expand_as(cos_sim)  # Expand to [B, H*W]

#     # Validate dimensions before computing loss
#     assert cos_sim.shape == mask.shape, f"Shape mismatch: cos_sim {cos_sim.shape}, mask {mask.shape}"

#     # Compute loss
#     loss = (1 - cos_sim) * mask  # Apply mask
#     loss = loss.sum() / mask.sum()  # Normalize by active regions in mask
#     return loss


# def imitation_focal_loss_err(teacher, student, mask, gamma=2.0):
#     """
#     Focal loss-based imitation loss for knowledge distillation.
#     Args:
#         teacher: Teacher model's output (logits or features).
#         student: Student model's output (logits or features).
#         mask: Mask to focus on specific regions.
#         gamma: Focusing parameter for focal loss.
#     Returns:
#         Focal imitation loss value.
#     """
#     if student is None or teacher is None:
#         return 0

#     # Compute per-pixel probability
#     teacher_prob = F.softmax(teacher, dim=-1)
#     student_prob = F.softmax(student, dim=-1)

#     # Compute focal loss
#     focal_loss = -teacher_prob * (1 - student_prob) ** gamma * torch.log(student_prob + 1e-8)
#     loss = (focal_loss.sum(dim=-1) * mask).sum() / mask.sum()
#     return loss




# def imitation_focal_loss(teacher, student, mask, alpha=0.25, gamma=2.0):
#     """
#     Focal loss for imitation in knowledge distillation.
#     Args:
#         teacher: Teacher model's output (logits or features).
#         student: Student model's output (logits or features).
#         mask: Mask to focus on specific regions.
#         alpha: Weight for positive samples in focal loss.
#         gamma: Modulation factor for hard-to-classify samples.
#     Returns:
#         Focal loss value.
#     """
#     if student is None or teacher is None:
#         return 0

#     # Align spatial dimensions of teacher and student outputs
#     if teacher.shape[-2:] != student.shape[-2:]:
#         teacher = F.interpolate(teacher, size=student.shape[-2:], mode='bilinear', align_corners=False)

#     # Handle mask dimensions
#     if len(mask.shape) > 2:  # If mask is not [batch_size, spatial_size]
#         print(f"Reshaping mask from shape: {mask.shape}")
#         mask = mask.view(mask.shape[0], -1)  # Flatten spatial dimensions
    
#     batch_size, h_w = mask.shape  # Unpack after reshaping

#     # Reshape and resize the mask to match student spatial dimensions
#     feature_h, feature_w = student.shape[-2:]  # E.g., 128 positions
#     mask = mask.view(batch_size, 1, int(h_w ** 0.5), int(h_w ** 0.5))  # Reshape to [B, 1, H, W]
#     mask = F.interpolate(mask, size=(feature_h, feature_w), mode='nearest')  # Resize to match feature dimensions
#     mask = mask.view(batch_size, -1)  # Flatten back to [B, H*W]

#     # Compute softmax probabilities
#     prob_teacher = F.softmax(teacher, dim=-1)
#     prob_student = F.softmax(student, dim=-1)

#     # Compute focal loss
#     focal_loss = -alpha * (1 - prob_student) ** gamma * prob_teacher * torch.log(prob_student + 1e-6)

#     # Validate dimensions
#     assert focal_loss.sum(dim=-1).shape == mask.shape, f"Shape mismatch: focal_loss {focal_loss.sum(dim=-1).shape}, mask {mask.shape}"

#     # Apply mask and compute normalized loss
#     loss = (focal_loss.sum(dim=-1) * mask).sum() / mask.sum()
#     return loss

## this function is  not working and give error


def attention_transfer_loss(teacher, student, mask):
    """
    Attention transfer loss for knowledge distillation.
    Args:
        teacher: Teacher model's feature maps (intermediate outputs).
        student: Student model's feature maps (intermediate outputs).
        mask: Mask to focus on specific regions.
    Returns:
        Attention transfer loss value.
    """
    if student is None or teacher is None:
        return 0

    # Compute attention maps
    teacher_attention = torch.norm(teacher, p=2, dim=1, keepdim=True)  # L2 norm along channel axis
    student_attention = torch.norm(student, p=2, dim=1, keepdim=True)

    # Normalize attention maps
    teacher_attention = F.normalize(teacher_attention, p=2, dim=(2, 3))
    student_attention = F.normalize(student_attention, p=2, dim=(2, 3))

    # Compute loss
    loss = torch.pow(teacher_attention - student_attention, 2) * mask
    loss = loss.sum() / mask.sum()
    return loss



# ----------------------------------------


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets, teacher=None, student=None, mask=None):  # predictions, targets, model
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        # lmask = imitation_loss(teacher, student, mask) * 0.01
        lmask = imitation_loss_kl(teacher, student, mask)*0.01
        # lmask = combined_imitation_loss(teacher, student, mask)*0.01
        # lmask = imitation_loss_smooth_l1(teacher, student, mask)*0.01
        # lmask = imitation_loss_smooth_l1(teacher, student, mask)*0.09
        # lmask = cosine_similarity_loss(teacher=teacher, student=student, mask=mask)*0.01
        # lmask = imitation_focal_loss(teacher=teacher, student=student, mask=mask)*0.01
        # lmask = attention_transfer_loss(teacher=teacher, student=student, mask=mask)*0.01

        return (lbox + lobj + lcls + lmask) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

from .model import BaseModel


class Discriminator(BaseModel):
    def metric_run(self, input_imgs, labels):
        loss, logits = self.forward(input_imgs=input_imgs, labels=labels)[:2]
        predictions = logits.max(dim=1)[1].cpu()
        acc = (predictions == labels).sum().float() / input_imgs.shape[0]
        return loss, {'acc': acc}

from .model import BaseModel


class Discriminator(BaseModel):
    def metric_run(self, input_ids, attention_mask, labels):
        loss, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)[:2]
        predictions = logits.max(dim=1)[1].cpu()
        acc = (predictions == labels).sum().float() / input_ids.shape[0]
        return loss, acc

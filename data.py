from pathlib import Path

from torchtune.datasets import SFTDataset
from torchtune.data._messages import Message
from torchtune.modules.transforms import Transform


class ARCToMessages(Transform):

    def __init__(
        self,
        new_system_prompt = None,
        image_dir="",
        image_tag="",
        train_on_examples=False,
        inference=False,
    ):
        self.new_system_prompt = new_system_prompt
        self.train_on_examples = train_on_examples
        self.inference = inference

    def __call__(self, sample):
        # Dataset images to be prepended to the first user message
        # img_content = []
        # for img in sample[self._column_map["images"]]:
        #     img_content.append({"type": "image", "content": img})

        messages = []
        if self.new_system_prompt is not None:
            content = self.new_system_prompt
            message = Message(role="system", content=content, masked=True, eot=True)
            messages.append(message)

        # Examples to messages
        mask_examples = not self.train_on_examples
        for i, data in enumerate(sample["train"]):
            content = [{"type": "text", "content": str(data["input"])}]
            message = Message(role="user", content=content, masked=True)
            messages.append(message)

            content = [{"type": "text", "content": str(data["output"])}]
            message = Message(role="assistant", content=content, masked=mask_examples)
            messages.append(message)

        # Test message
        data = sample["test"][0]
        content = [{"type": "text", "content": str(data["input"])}]
        message = Message(role="user", content=content, masked=True)
        messages.append(message)

        if not self.inference:
            content = [{"type": "text", "content": str(data["output"])}]
            message = Message(role="assistant", content=content, masked=False)
            messages.append(message)
        else:
            messages.append(Message(role="assistant", content=""))

        return {"messages": messages}


def arc_dataset(
    model_transform,
    *,
    source,
    split,
    new_system_prompt = None,
    train_on_examples=False,
    image_tag = None,
    image_dir = "",
    **load_dataset_kwargs,
):

    message_transform = ARCToMessages(
        new_system_prompt=new_system_prompt,
        train_on_examples=train_on_examples,
        image_dir=Path(image_dir),
        image_tag=image_tag,
    )

    ds = SFTDataset(
        source=source,
        split=split,
        message_transform=message_transform,
        model_transform=model_transform,
        **load_dataset_kwargs,
    )

    return ds

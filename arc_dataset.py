# from torchtune.datasets import SFTDataset

# class ARCDataset(SFTDataset):
#     """
#     Dataset for ARC dataset. This dataset is used for supervised fine-tuning (SFT) of LLMs.
#     The dataset is a collection of question and answer pairs from the ARC dataset.
#     Each sample in the dataset is a dictionary containing the following keys:
#     - "t": the question to be answered
#     - "answer": the correct answer to the question
#     - "choices": a list of possible answers to the question
#     """
#     def __init__(
#         self,
#         *,
#         source: str,
#         message_transform: Transform,
#         model_transform: Transform,
#         filter_fn: Optional[Callable] = None,
#         **load_dataset_kwargs: Dict[str, Any],
#     ) -> None:

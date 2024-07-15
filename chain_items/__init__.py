from langchain_core.runnables import RunnableSerializable
from langchain_core.prompt_values import ChatPromptValue
from abc import ABC

class PromptValueToChatML (RunnableSerializable[ChatPromptValue, list[dict[str, str]]], ABC):
  type_role_map = {
    'system': 'system',
    'human': 'user',
    'ai': 'assistant'
  }

  def invoke (self, input: ChatPromptValue, config):
    print('input', input)
    ret = [{'role': self.type_role_map[message.type], 'content': message.content} for message in input.messages]
    # print(f'PromptValueToChatML', ret)
    return ret

from langchain_core.runnables import RunnableSerializable
from transformers import PreTrainedTokenizer
from abc import ABC
from pydantic import Field

class ChatTemplateApplier (RunnableSerializable[list[dict[str, str]], str], ABC):
  tokenizer: PreTrainedTokenizer = Field(...)

  class Config:
    arbitrary_types_allowed = True
    extra = "allow"

  def __init__ (self, tokenizer: PreTrainedTokenizer):
    super().__init__()
    self.tokenizer = tokenizer

  def invoke (self, input: list[dict[str, str]], config):
    ret = self.tokenizer.apply_chat_template(input, tokenize=False, return_full_text=False, max_length=1024, add_generation_prompt=True)
    # print(f'ChatTemplateApplier', ret)
    return ret

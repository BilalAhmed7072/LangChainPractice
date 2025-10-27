from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#import os
#os.environ['HF_HOME'] = 'D:/huggingface_cache'

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(model=model_name,
                tokenizer=tokenizer,
                max_new_tokens=200,
                temperature= 0.7,
                top_p=0.9)
llm_local = HuggingFacePipeline(pipeline=pipe)

prompt = "what is role of AI in these days?"
result = llm_local.invoke(prompt)
print(result.content)
                                             

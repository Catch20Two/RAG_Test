

from langchain.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import Chroma
import nest_asyncio
#from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnablePassthrough
import sys, os


# Initiating the LLM and config options
n_gpu_layers = 1  # Metal set to 1
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_name='sentence-transformers/all-mpnet-base-v2'
persist_directory="./chroma_db1"


model = LlamaCpp(
	model_path="./LLM/llama-2-7b-chat.Q5_0.gguf",
	n_gpu_layers=n_gpu_layers,
	n_batch=n_batch,
	n_ctx=2048,
	f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
	callback_manager=callback_manager,
	verbose=True,
)

nest_asyncio.apply()
# Articles to index
articles = [
		   "https://en.wikipedia.org/wiki/Punk_rock"
		#     "https://www.medicalnewstoday.com/human-biology/",
		#    "https://www.trivianerd.com/topic/human-body-trivia/",
		#    "https://www.watercoolertrivia.com/trivia-questions/anatomy-trivia-questions"
 ]

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def build_DB(wiki_pages,num):
	# Scrapes the blogs above
	loader = AsyncChromiumLoader(wiki_pages)
	docs = loader.load()
	print("Web scrapping wiki pages")

	# Converts HTML to plain text 
	html2text = Html2TextTransformer()
	docs_transformed = html2text.transform_documents(docs)
	print("Converting HTML to text")

	# Chunk text
	text_splitter = CharacterTextSplitter(chunk_size = 128, chunk_overlap = 0)


	chunked_documents = text_splitter.split_documents(docs_transformed)


	print("Chunking text")
	# Load chunked documents into the FAISS index
	db = Chroma.from_documents(chunked_documents, 
							HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'),
							persist_directory='./chroma_db' + str(num),
							collection_name='v_db')
	
	with open('./chroma_db' + str(num) + '/chunked_documents.txt', 'w') as f:
		f.write(str(chunked_documents))
	return db

def add_to_DB(wiki_pages, db1):
	db2 = build_DB(wiki_pages, 2)

	db2_data=db2._collection.get(include=['documents','metadatas','embeddings'])

	db1._collection.add(
		embeddings=db2_data['embeddings'],
		metadatas=db2_data['metadatas'],
		documents=db2_data['documents'],
		ids=db2_data['ids']
	)

def add_articles(articles):
	new_article = input("\nAdd a new article to reference: ")
	articles.append(new_article)
	return articles

def add_wiki(articles):
	new_wiki = input("\nWhat wikipedia page should be added to the articles?: ")
	articles.append("https://en.wikipedia.org/wiki/" + new_wiki)
	print("https://en.wikipedia.org/wiki/" + new_wiki)
	return articles

def remove_articles(articles):
	c = 0
	for i in articles:
		print(str(c) + " " + i)
		c += 1
	remove = input("\nInput number of article to remove: ")
	if len(articles) > int(remove) and int(remove) > 0:
		#articles.pop(remove)
		del articles[int(remove)]
	return articles

def update(articles, db1):
	add_to_DB(articles, db1)
	retriever = db1.as_retriever()
	return retriever


def main(articles):
	# Set up the initial Chroma Database and retriver 
	db1 = build_DB(articles, 1)
	retriever = db1.as_retriever()

	# Define prompt
	p = 'Answer the question with a single sentence.'
	prompt_template= "### [INST] Instruction: " + p + " Here is context to help:{context}### QUESTION:{question}[/INST]"

	# Abstraction of Prompt
	prompt = ChatPromptTemplate.from_template(prompt_template)
	output_parser = StrOutputParser()

	# Creating an LLM Chain 
	llm_chain = LLMChain(llm=model, prompt=prompt)

	# RAG Chain
	rag_chain = ( 
	{"context": retriever, "question": RunnablePassthrough()}
		| llm_chain
	)

	print(" ---- Type 'help' for options ----")
	while True:
		query = input("\nEnter a query or command: ")
		# Define and run queries and commends
		if query == "exit":
			break
		if query == "add":
			articles = add_articles(articles)
			print(articles)
			continue
		if query == "wiki":
			articles = add_wiki(articles)
			print(articles)
			continue
		if query == "remove":
			articles = remove_articles(articles)
			print(articles)
			continue
		if query == "update":
			update(articles, db1)
			print("Updated DB with current articles list")
			continue
		if query == "prompt":
			p = input("\nEnter a new prompt for the query: ")
			print(p)
			continue
		if query.strip() == "":
			print(p)
			continue

		# Update all of the prompt info
		retriever = db1.as_retriever()
		prompt_template= "### [INST] Instruction: " + p + " Here is context to help:{context}### QUESTION:{question}[/INST]"

		# Abstraction of Prompt
		prompt = ChatPromptTemplate.from_template(prompt_template)

		# Creating an LLM Chain 
		llm_chain = LLMChain(llm=model, prompt=prompt)

		rag_chain = ( 
		{"context": retriever, "question": RunnablePassthrough()}
			| llm_chain
			#| StrOutputParser()
		)
		print(prompt_template)
		rag_chain.invoke(query)

if __name__ == "__main__":
	main(articles)

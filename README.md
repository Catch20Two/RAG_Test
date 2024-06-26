
This is a very simple example of how to implement Retrieval-Augmented Generation (RAG) and a localized Large Language Model (LLM). In theroy, a local LLM can operate without connecting to the internet or centalized AI (e.g. OpenAI, ChatGPT, etc). This example will need the internet only for the purpose of adding new material to its knowledge corpus. RAG allows for the LLM to only concern itself with the provided corpus and implements the option to structure the output using a user defined template. The ultimate goal is with this approach is to acheive a secure way to review documents without sharing any information with unwanted parties. The "RAG_Test.py" was developed and tested within a Ubuntu virtual machine with the intent of running a more sophisticated version in an AWS EC2 instance.

Getting Started --------------------------------------------------------------------------------------------------------------------------------------------
Create the folder called "LLM". Then download and add the following LLM from HuggingFace (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF):
llama-2-7b-chat.Q5_0.gguf

Running the code will create two additional folders i.e. "chroma_db1" and "chroma_db2". 

1. Start the program. A new local Chroma database ("chroma_db1") is created fresh each time. This was intentional for demo purposes and can be changed by adding a function to check for a persistent Chroma db.
2. The LLM will need to initialize and a starting knowledge db will be built using "https://en.wikipedia.org/wiki/Punk_rock". This is done with various Langchain tools which scrape the HTML text from the website and split the text into sentence-transformed weights
for the first Chroma db. This should take about a minute depending on your local processing power. 
3. Using the program:
   You will be prompted with "Enter a query or command:" the following options exist:
   1. "exit" - Ends the program.
   2. "add" - Allows for new web pages to be added. Note: the option, "update" will need to be used to update the db with the new web page.
   3. "wiki" - Rather than typing out a whole URL, you can just type in a word affiliated with a wikipedia page to add to the list of web pages i.e. "https://en.wikipedia.org/wiki/" + your word
   4. "remove" - Delete a webpage from the list. Note: "update" will still need to be used.
   5. "update" - Refreshes the Chroma dbs with the latest list of web pages.
   6. "prompt" - Allows for a new RAG prompt to be created by the user. The default is "Answer the question with a single sentence" which will attempt to do just that. Changing this will result in different output and processing times. Note: some fun options are "Answer the question with a single word" or "Answer the question in the form of a question".
  
      Anything other than the options listed above will run a query against the corpus of knowledge built within the Chroma db using the very basic Llama2-7B LLM to make sense of it. 

  



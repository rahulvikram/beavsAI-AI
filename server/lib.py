# DONT SPAM run this code a lot because it uses tom's openAI key and it will make him broke
import os

import pinecone
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

load_dotenv()

# constants
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]

METRIC='cosine'
DIMENSIONS=1536

# accept userinput for filename (based on the class they want to see)
course_code = input("Enter CS course code (Number only, e.g. '271' rather than 'CS271'): ")
filename = f"CS{course_code}" # turns e.g. 161 into CS161

# placeholder message while PDF is loading
print(f"\n Loading PDF... \n")

# load PDF via user input provided for filename
loader = UnstructuredPDFLoader(f"../data/{filename}.pdf")
data = loader.load()

# if successful, print success messages
print(f"\n Loaded {len(data)} documents \n")
print(f"\n There are {len(data[0].page_content)} characters in the document \n")

# print filler text while splitting data
print(f"\n Splitting...\n")

# Split the data into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# if successful, print success message
print(f"\n Successsfully Split. You now have {len(texts)} documents \n")

# Create embeddings of the documents to get ready for semantic search
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at https://www.pinecone.io/
    environment=PINECONE_API_ENV,  # find next to the API key
)

# dynamically set index name based on user input
index_name = f"cs{course_code}"

# check if an index already exists, if it does, delete and replace with a new index to avoid the premium plan charges
# unfortunately this is the only way to do this

if len(pinecone.list_indexes()) == 1: # IF an index already does exist
    # if the index we want to add matches the index already existing
    if index_name == pinecone.list_indexes()[0]:
        pass; # do nothing, move onto PDF loading
    else: # if the index existing does not match the one we want,
        pinecone.delete_index(pinecone.list_indexes()[0]) # deletes existing index
        pinecone.create_index(name=index_name, metric=METRIC, dimension=DIMENSIONS) # creates new index based on what we want
else: # IF an index does NOT yet exist
    pinecone.create_index(name=index_name, metric=METRIC, dimension=DIMENSIONS) # creates new index based on what we want

# load the PDF into the index we created
docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# initializes query variable
query = ''
while query != 'exit':
    # takes user input for the query
    query = input("enter your query (type 'exit' to quit): ")

    # checks to make sure the query isn't exit or blank
    # if it is blank, redo the query process; if it is exit, quit
    if query.replace(" ", "") != '' and query != 'exit':

        # conduct a similarity search on the index based on user query
        docs = docsearch.similarity_search(query)

        # --------- Get a natural language answer to the question --------- #
        # initialize the OpenAI GPT-3 model
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

        # creates chain 
        chain = load_qa_chain(llm, chain_type="stuff")

        # retrieves a response based on user query using chain
        llm_response = chain.run(input_documents=docs, question=query)

        # prints the response to the terminal
        print(llm_response) # print the response returned

        # ----------------------------------------------------------------- #

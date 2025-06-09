#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INDEXING STAGE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# importing the necessary modules
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv


# set up the rag_chatbot class
class rag_chatbot:
    load_dotenv()
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        # defining the web_paths with the list of webpage urls to scrape
        web_paths = [
            "https://www.kaspersky.com/resource-center/definitions/what-is-cryptocurrency",
            "https://www.kaspersky.com/resource-center/definitions/what-is-bitcoin",
            "https://www.kaspersky.com/resource-center/definitions/what-is-crypto-mining",
            "https://www.kaspersky.com/resource-center/preemptive-safety/guide-to-cryptocurrency-safety",
            "https://www.kaspersky.com/resource-center/preemptive-safety/how-to-avoid-nft-scams"
        ]
        # initializing with the self and calling the methods in the rag class 
        
        self.load_doc = self.web_loader(web_paths)
        self.split_doc = self.document_splitter(self.load_doc)
        self.store_document = self.vector_store(self.split_doc)
        self.rag_prompt = self.prompt_template()
        self.rag_chain = self.crypto_rag_chain(self.store_document, self.rag_prompt, self.format_doc)
        
    # defining the methods in full
        # the web_loader method
    def web_loader(self, web_paths):
            loader = WebBaseLoader(web_paths)
            data = loader.load()
            return data
        
        # the document loader method
           
    def document_splitter(self, data ):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200
            )
            chunks = splitter.split_documents(data)
            return chunks

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>RETRIEVAL STAGE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # the vectorstore method
        
    def vector_store(self, chunks):
            store = FAISS.from_documents(chunks, self.embeddings)
            vector_retriever = store.as_retriever(
            search_kwargs ={
                    "k": 5},
            search_type = "similarity"
            )
            return vector_retriever

    def sparse_retriever(self, chunks):
            bm_retriever = BM25Retriever(
                document = chunks, k = 10,
            )
            return bm_retriever
        
    def hybrid_retriever( self, vector_retriever, sparse_retriever):
            retrievers = [vector_retriever, sparse_retriever ]
            return retrievers
        
            
        # the prompt template
        
    def prompt_template(self):
            chat_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    # prompt engineering
                    """You are a crypto expert with a long standing experience about the crypto space,
                    your role as a virtual crypto assistant is to answer questions about crypto currency and the 
                    web3 space at large using the provided context delimited by tripple back ticks,
                    remember your role as a crypto analyst, you are to answer questions based on the provided
                    context, be nice to the users and if you do not know the answer to any question 
                    just say you do not and provide the contact 09014299504, to the users for further information,
                    avoid hallucinations, and be brief, provide references if you have to. 
                    context : ```{context}```
                    """
                ),
                HumanMessagePromptTemplate.from_template("{user_query}")
            ])
            return chat_template
        
 # creating the format function to render the retreived vector context in a string format
     
    def format_doc(self, chunks):
            string_format = "\n\n".join(chunk.page_content for chunk in chunks)
            return string_format 
            
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> GENERATION STAGE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
    
        # defining the rag chains first from document and second from source
        
    def crypto_rag_chain(self, hybrid_retriever, prompt_template, format_doc):
        rag_from_doc = RunnablePassthrough().assign(
# using the assign method of the RunnablePassthrough() to assign the context using the python lammda function to the format_doc function
            context=lambda x: format_doc(x["context"])
# hydrating the prompt by passing the formatted vector context into the prompt using the pipe | operator to chain the operation
        ) | prompt_template | self.llm | StrOutputParser()

        rag_from_source = RunnableParallel(
            {
                "context": lambda x: hybrid_retriever.invoke(x["user_query"]),
                "user_query": RunnablePassthrough()
            }
        ).assign(answer=rag_from_doc)
        return rag_from_source
        
        # To run the rag system
        
    def query(self, user_query: str) -> str:
            try:
                result = self.rag_chain.invoke({"user_query": user_query})
                # Handle both dict and object result
                if hasattr(result, "answer"):
                    return result.answer 
                elif isinstance(result, dict) and "answer" in result:
                    return result["answer"]
                else:
                    return str(result)
            except Exception as e:
                return f"Error in RAG query: {e}"
                      
        
        
from openai import AzureOpenAI
import os


class LLM:
    """Handles interactions with the Azure OpenAI LLM (Large Language Model).

    Attributes:
        client (AzureOpenAI): The Azure OpenAI client instance.
        model_name (str): The name of the Azure OpenAI LLM model to use.

    Methods:
        get_response(history, context, user_input): Generates a response from the LLM based on the conversation history, context, and user input.
    """
    def __init__(self):
        """Initializes the LLM class with Azure OpenAI client and model information."""
        # AzureOpenAI client setup
        azure_endpoint = os.getenv("AZURE_LLM_ENDPOINT")
        azure_deployment = os.getenv("AZURE_LLM_DEPLOYMENT_NAME")
        api_key = os.getenv("AZURE_LLM_API_KEY")
        api_version = os.getenv("AZURE_LLM_API_VERSION")

        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key
        )
        self.model_name = os.getenv("AZURE_LLM_MODEL_NAME")

    def get_response(self, history, context, user_input):
        """Generates a response from the LLM.

        Args:
            history (list): A list of previous messages in the conversation history.
            context (str): Relevant information from the knowledge base to provide context to the LLM.
            user_input (str): The user's current input.

        Returns:
            str: The LLM's generated response.
        """
        # Prepare the messages for the chat/completions endpoint
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if history:
            messages.extend(history)  # Add previous messages
        messages.append({"role": "user", "content": f"Context:\n{context}\n\n{user_input}"})

        #print("context:", context)
        print("user_input:", user_input)

        try:
            # Call Azure OpenAI API to generate a response
            response = self.client.chat.completions.create(
                model=self.model_name,  # Use 'engine' instead of 'model' for Azure OpenAI
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.6,
            )
            # Extract the response content
            print(response.choices[0].message.content)
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error generating a response."

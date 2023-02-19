
def prompt(func, model, tokenizer, msg="\nPlease enter your question: "):
    while True:
        question = input(msg)
        if question == "quit":
            break
        else:
            func(question, model, tokenizer)

def prompt_conversation(func, chatbot, msg="\nPlease enter your question: "):
    while True:
        question = input(msg)
        if question == "quit":
            break
        else:
            func(question, chatbot)
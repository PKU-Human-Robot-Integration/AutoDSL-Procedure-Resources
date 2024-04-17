def token_count_decorator(func):
    '''
        Token count decorator.

        @Arguments:
            func (function): The function to be decorated.

        @Returns:
            The result of the decorated function.

        @Functionality:
            This decorator calculates the number of tokens used in the function call, updates the token count and fee in a file, and adjusts the fee based on the model used. It then returns the result of the function call.
    '''

    def wrapper(*args, **kwargs):
        args_with_spaces = sum([str(arg).count(' ') for arg in args])
        kwargs_with_spaces = sum([str(value).count(' ') for key, value in kwargs.items()])
        token_num = args_with_spaces + kwargs_with_spaces
        result = func(*args, **kwargs)
        token_count_path = "data/token_count.txt"

        with open(token_count_path, 'r') as file:
            tot_token_num = int(file.readline())
            fee = float(file.readline())
        
        model_value = kwargs.get('model', "gpt-3.5-turbo")
        if model_value == "gpt-3.5-turbo":
            tot_token_num += token_num
            fee = fee + (0.002 * token_num / 1000.0)
        else:
            tot_token_num += token_num
            fee = fee + (0.002 * token_num / 100.0)
        with open(token_count_path, 'w') as file:
            file.write(str(tot_token_num) + '\n')
            file.write(str(fee) + '\n')
        return result
    
    return wrapper
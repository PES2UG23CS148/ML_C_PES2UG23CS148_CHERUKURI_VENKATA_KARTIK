# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    col=[t[-1].tolist() for t in tensor]
    def CalculateEntropy(param):
        return (-torch.sum(param*torch.log2(param)))
    label=list(set(col))
    list_probs=[]
    for x in label:
        list_probs.append(col.count(x)/len(col))
    k = (CalculateEntropy(torch.tensor(list_probs))).item()
    return k
    # TODO: Implement this function
    raise NotImplementedError
def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.
    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    cur_column_attribute = [t[attribute].tolist() for t in tensor]     
    multiple_class = list(set(cur_column_attribute))              
    label_col= [t[-1].tolist() for t in tensor]                    
    total_length_of_column=len(cur_column_attribute)
    o=0
    for x in multiple_class:
        feature_count = cur_column_attribute.count(x)          
        mul_factor_probs = torch.tensor(feature_count/total_length_of_column)
        t1=[]
        t2=[]
        for i in range(len(cur_column_attribute)):
            if cur_column_attribute[i]==x:
                t1.append(x)                    # t1 contains column attributes which a unique in each iteration
                t2.append(label_col[i])      # t2 contains the corresponding target column which is mapped to the attribute column
        new_tensor = torch.cat((torch.tensor(t1).unsqueeze(1),torch.tensor(t2).unsqueeze(1)),dim=1)     
        test_entropy = get_entropy_of_dataset(new_tensor)
        if torch.isnan(torch.tensor(test_entropy))==False:
            o+=(mul_factor_probs*test_entropy).item()     
    return o
    pass
    # TODO: Implement this function
    raise NotImplementedError
def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)
    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.
    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    return (torch.round(torch.tensor(get_entropy_of_dataset(tensor)) - get_avg_info_of_attribute(tensor,attribute),decimals=4)).item() 
    pass

    # TODO: Implement this function
    raise NotImplementedError
def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.
    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)
    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    gain_dictionary={}
    for i in range(len(tensor[0]) - 1):
        gain_dictionary[i]=get_information_gain(tensor,i)           
    maximumgain = max(gain_dictionary.values())
    for i in gain_dictionary.keys():
        if gain_dictionary[i] == maximumgain:
            return (gain_dictionary,int(i))                         
    return ({},-1)
    pass
    # TODO: Implement this function
    raise NotImplementedError

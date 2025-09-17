def find_duplicates(arr):
    """
    Find duplicate elements in an array.
    
    Args:
        arr: List of elements to check for duplicates
        
    Returns:
        List of duplicate elements found
    """
    seen = {}
    duplicates = []
    
    for item in arr:
        if item in seen:
            if seen[item] == 1:
                duplicates.append(item)
            seen[item] += 1
        else:
            seen[item] = 1
            
    return duplicates
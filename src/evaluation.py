import numpy as np

def calculate_profit(y_true, y_pred_proba, threshold=0.333, num_vars=0, debug=False):
    """
    Calculates the business profit (in EUR) for the marketing campaign.
    
    Parameters:
    y_true: True labels (0 or 1) indicating if the customer takes the offer
    y_pred_proba: Probability of class 1 returned by the model
    threshold: Cut-off threshold (probability above which we send the offer)
    num_vars: Number of variables used in the model (penalty of -200 EUR each)
    debug: If True, prints out the campaign details

    Formula:
    Profit = (TP * 10) - (FP * 5) - (num_vars * 200)
    Where:
    TP (True Positives) - selected customers who took the offer
    FP (False Positives) - selected customers who did not take the offer
    max 1000 offers can be sent. In the case of higher number of indexes, only first 1000 ids are assessed. 
    """
    y_true = np.array(y_true).ravel()
    y_pred_proba = np.array(y_pred_proba).ravel()
    
    eligible_indices = np.where(y_pred_proba > threshold)[0]
    
    # Business constraint: sending offer to max 1000 people
    selected_indices = eligible_indices[:1000] if len(eligible_indices) > 1000 else eligible_indices

    if len(selected_indices) == 0:
        TP, FP = 0, 0
    else:
        TP = np.sum(y_true[selected_indices] == 1)
        FP = np.sum(y_true[selected_indices] == 0)
        
    profit_from_customers = (TP * 10) - (FP * 5)
    cost_of_variables = num_vars * 200
    total_score = profit_from_customers - cost_of_variables
    
    if debug:
        print(f"Selected customers for the campaign: {len(selected_indices)}")
        print(f"Hits (TP): {TP} (+{TP*10} EUR)")
        print(f"Misses (FP): {FP} (-{FP*5} EUR)")
        print(f"Cost of variables ({num_vars} used): -{cost_of_variables} EUR")
        print(f"FINAL SCORE: {total_score} EUR\n")
        
    return total_score
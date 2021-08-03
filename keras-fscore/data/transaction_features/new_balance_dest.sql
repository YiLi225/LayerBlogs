/*
    Fraud Detection Project Example
    `new_balance_dest` Feature

    Since the destination account balances being zero is a strong indicator
    of fraud, we replace the value of 0 with -1 which will be more useful to
    a suitable machine-learning (ML) algorithm detecting fraud.
*/

SELECT transactionId,
       if(oldbalanceDest == 0 and newbalanceDest == 0 and amount != 0, -1,
          newbalanceDest) as new_balance_dest
FROM transactions

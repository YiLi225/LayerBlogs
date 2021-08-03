/*
    Fraud Detection Project Example
    `old_balance_orig` Feature

    Since the originating account balances being zero is a strong indicator
    of fraud, we replace the value of 0 with -1 which will be more useful to
    a suitable machine-learning (ML) algorithm detecting fraud.

*/

SELECT transactionId,
       if(oldbalanceOrg == 0 and newbalanceOrig == 0 and amount != 0, -1,
          oldbalanceDest) as old_balance_orig
FROM transactions

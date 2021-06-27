/*
    Fraud Detection Project Example
    `errorBalanceDest` Feature

    Motivated by the possibility of zero-balances serving to differentiate
    between fraudulent and genuine transactions, we add a new feature recording
    errors in the destination accounts per transaction.

*/

SELECT transactionId,
       oldbalanceDest + amount - newbalanceDest as errorBalanceDest
FROM transactions

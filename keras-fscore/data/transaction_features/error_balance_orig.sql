/*
    Fraud Detection Project Example
    `errorBalanceOrig` Feature

    Motivated by the possibility of zero-balances serving to differentiate
    between fraudulent and genuine transactions, we add a new feature recording
    errors in the originating accounts per transaction.

*/

SELECT transactionId,
       newbalanceOrig + amount - oldbalanceOrg as errorBalanceOrig
FROM transactions

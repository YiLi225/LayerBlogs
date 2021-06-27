/*
    Fraud Detection Project Example
    `is_fraud` Feature

    This will be our label to pass to our model
*/

SELECT transactionId,
       isFraud as is_fraud
FROM transactions

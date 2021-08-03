/*
    Fraud Detection Project Example
    `type` Feature

*/

SELECT transactionId,
       CASE
           WHEN type = "TRANSFER" THEN 0
           WHEN type = "CASH_OUT" THEN 1
           WHEN type = "CASH_IN" THEN 2
           WHEN type = "DEBIT" THEN 3
           WHEN type = "PAYMENT" THEN 4
           ELSE -1
           END
           as type
FROM transactions
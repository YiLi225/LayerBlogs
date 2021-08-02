/*
    Titanic Survival Project Example
    `FareBand` Feature
*/

SELECT PassengerId,
       CASE
           WHEN Fare <= 7.91 THEN 0
           WHEN Fare > 7.91 and Fare <= 14.454 THEN 1
           WHEN Fare > 14.454 and Fare <= 31 THEN 2
           WHEN Fare > 31 THEN 3
           END as FareBand
FROM titanic
/*
    Titanic Survival Project Example
    `Sex` Feature
*/

SELECT PassengerId,
       CASE
           WHEN sex = "female" THEN 1
           WHEN sex = "male" THEN 0
           END
           as Sex
FROM titanic
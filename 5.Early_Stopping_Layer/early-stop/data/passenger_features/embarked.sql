/*
    Titanic Survival Project Example
    `EmbarkStatus` Feature
*/

SELECT PassengerId,
       CASE
           WHEN Embarked = "S" THEN 0
           WHEN Embarked = "C" THEN 1
           WHEN Embarked = "Q" THEN 2
           ELSE -1
           END
           as EmbarkStatus
FROM titanic
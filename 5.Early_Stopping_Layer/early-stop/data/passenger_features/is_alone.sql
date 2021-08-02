/*
    Titanic Survival Project Example
    `IsAlone` Feature
*/

WITH familySizeDF as (
    SELECT PassengerId,
           SibSp + Parch + 1 as familySize
    FROM titanic
)

SELECT PassengerId,
       CASE
           WHEN familySize = 1
               THEN 1
           ELSE 0
           END
           AS IsAlone
FROM familySizeDF
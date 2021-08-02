/*
    Titanic Survival Project Example
    `AgeBand` Feature
*/

WITH ageDF as (SELECT PassengerId,
                      CASE
                          WHEN age is null
                              THEN m.AverageAge
                          ELSE age
                          END
                          as AverageAge
               FROM titanic t
                        LEFT JOIN (
                   SELECT Pclass,
                          Sex,
                          avg(Age) as AverageAge
                   FROM titanic
                   GROUP BY 1, 2
               ) m on m.Pclass = t.Pclass and m.Sex = t.Sex)

SELECT PassengerId,
       CASE
           WHEN AverageAge <= 16 THEN 0
           WHEN AverageAge > 16 and AverageAge <= 32 THEN 1
           WHEN AverageAge > 32 and AverageAge <= 48 THEN 2
           WHEN AverageAge > 48 and AverageAge <= 64 THEN 3
           WHEN AverageAge > 64 THEN 4
           END as AgeBand
FROM ageDF
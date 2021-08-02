/*
    Titanic Survival Project Example
    `Title` Feature
*/

WITH titleDF as (
    SELECT PassengerId,
           regexp_replace(
               regexp_replace(
                regexp_extract(Name, ' (\\\w+)\\\.',1)
                ,'^(Don|Countess|Col|Rev|Lady|Capt|Dr|Sir|Jonkheer|Major)$', 'Rare')
                ,'^(Mlle|Ms|Mme)$', 'Miss'
            )
         as parsedTitle
    FROM titanic
    )

SELECT PassengerId,
    CASE
        WHEN parsedTitle = "Mr" THEN 1
        WHEN parsedTitle = "Miss" THEN 2
        WHEN parsedTitle = "Mrs" THEN 3
        WHEN parsedTitle = "Master" THEN 4
        WHEN parsedTitle = "Rare" THEN 5
    END as Title
FROM titleDF
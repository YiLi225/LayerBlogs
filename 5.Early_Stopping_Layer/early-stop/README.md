# Early Stopping technique with Layer

A theoretical walk-through and code implementation of early stopping in Keras with Layer. 

## What we are going to learn?

- Feature Store: We are going to use SQL queries to build the `passenger` features.
- Load `passenger` features and use it to train the `early stopping model`
- Experimentation tracking
 - logging `accuracy` metric

## Installation & Running

To check out the Layer early stopping example, run:

```bash
layer init early-stop
```

To run the project:

```bash
layer run
```

## File Structure

```yaml
.
├── .layer
├── data
│   ├── passenger_features	        # feature definitions
│   │   ├── ageband.sql				# Age Band of the passenger
│   │   ├── embarked.sql  			# Embarked or not
│   │   ├── fareband.sql			# Fare Band of the passenger
│   │   ├── is_alone.sql			# Is Passenger travelling alone
│   │   ├── sex.sql					# Sex of the passenger
│   │   ├── survived.sql 			# Survived or not
│   │   ├── title.sql				# Title of the passenger
│   │   └── dataset.yml				# Declares the metadata of the features above
│   └── titanic_data
│       └── dataset.yml				# Declares where our source `titanic` dataset is
├── models
│   └── survival_model
│       ├── model.yml				# Training directives of our model
│       ├── model.py				# Source code of the `Survival` model
│       └── requirements.txt		# Environment config file
└── README.md
```

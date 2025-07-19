import pandas as pd
class TitanicPreprocessor:
    def __init__(self):
        pass
    
    #1. fill missing data before processing other columns
    def fill_missing_values(self, df):
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Cabin']=df['Cabin'].fillna('U') 
        return df

    def extract_titles(self, df):
        df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\s*\.')
        df["Title"] = df["Title"].replace({
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Dr': 'Other', 'Major': 'Other', 'Col': 'Other',
    'Lady': 'Miss', 'Sir': 'Mr', 'Capt': 'Other',
    'the Countess': 'Other', 'Don': 'Other', 'Jonkheer': 'Other', 'Rev': 'Other'
})
        return df

    def extract_deck(self, df):
        df['Cabin']=df['Cabin'].fillna('U')
        df['Deck']=df['Cabin'].astype(str).str[0]
        return df 
       
    
    def create_family_features(self, df):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['FamilyCategory'] = pd.cut(df['FamilySize'], bins=[0,1,4,8,11], labels=['Solo', 'Small', 'Medium', 'Large'])
        return df
    
    #5. drop name, sibsp, parch and cabin (used above) only after processing, not before. rest like pax id, ticket and fare are anyways useless 
    def drop_unnecessary(self, df):
        to_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin' ]
        df = df.drop(columns=to_drop)
        return df

    #6. to be done at the end, once you have all the newly made columns (do enconding at the end, cleaning first)
    def encode_categoricals(self, df):
        to_dummy = ['Title', 'Sex', 'Embarked', 'Deck', 'FamilyCategory']
        df = pd.get_dummies(df, columns=to_dummy, drop_first=True)
        return df

    def preprocess(self, df):
        df = self.fill_missing_values(df)
        df = self.extract_titles(df)
        df = self.extract_deck(df)
        df = self.create_family_features(df)
        df = self.drop_unnecessary(df)
        df = self.encode_categoricals(df)
        return df

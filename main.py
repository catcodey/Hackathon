import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from streamlit_option_menu import option_menu
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#1 is bad 0 is good
df = pd.read_csv('german_credit_data.csv')
df=df.drop("Unnamed: 0",axis=1)

df['Saving accounts'] = df['Saving accounts'].fillna(df['Saving accounts'].mode()[0])  # Categorical example
df['Credit amount'] = df['Credit amount'].fillna(df['Credit amount'].median())  # Numerical example

def determine_risk(row):
    risk_score = 0
    
    if row['Credit amount'] > 5000:  # credit is high
        risk_score += 1
    if row['Duration'] > 24:  # duration longer than 2 years
        risk_score += 1
    if str(row['Saving accounts']).lower() in ['little', 'nan']:  # low savings
        risk_score += 1
    if str(row['Checking account']).lower() in ['little', 'nan']:  # low checking
        risk_score += 1
    if str(row['Housing']).lower() == 'rent':  # renting
        risk_score += 1
    if row['Age'] < 25:  # young
        risk_score += 1
    if str(row['Purpose']).lower() in ['radio/TV', 'furniture/equipment']:  # luxury
        risk_score += 1
    
    # if risk score crosses a threshold, label as bad
    return 1 if risk_score >= 3 else 0


df['Risk'] = df.apply(determine_risk, axis=1)

with st.sidebar:
  selected=option_menu(
    menu_title=None,
    options=["overview","risk based assesment","credit amount vs duration","age based analysis","outlier analysis","Types of housing and savings analysis","ml model analysis"],
    default_index=0)

if selected=="overview":
    st.title("German Credit Risk Analysis")
    st.text(" This is the German Credit dataset")
    st.header("Dataframe")
    st.divider()
    st.write(df.head(5))
    st.write(" ")
  

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        show_info = st.button("***Show Column Info***")
        if show_info:

            col_info = pd.DataFrame({
                "Column": df.columns,
                "Non-Null Count": df.notnull().sum(),
                "Data Type": df.dtypes
            })
            st.write(col_info)

    with row1_col2:
        check_null = st.button("***Check Null Values***")
        if check_null:
            mis = df.isnull().sum()
            st.write(mis)

            if mis.sum() == 0:
                st.markdown(
                    '<p style="font-size: 20px; color: green;"><b>No missing values in any column.</b></p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<p style="font-size: 20px; color: red;"><b>Some columns have missing values.</b></p>',
                    unsafe_allow_html=True
                )

    with row2_col1:
        describe = st.button("***Describe Data***")
        if describe:
            st.dataframe(df.describe())

    with row2_col2:
        show_shape = st.button("***Show Shape***")
        if show_shape:
            st.markdown(
                f"<p style='font-size:20px;'>Dataset Shape: <b>{df.shape[0]} rows × {df.shape[1]} columns</b></p>",
                unsafe_allow_html=True
            )

if selected =="risk based assesment":
    st.title("Risk Assesment")
    st.divider()
    tab1, tab2, tab3 = st.tabs(["gender count based on risk","purpose vs risk", "credit vs risk"])

    with tab1:
        st.header("Gender Count Based on Risk")
        st.write("") 


        cross_tab = pd.crosstab(df['Sex'], df['Risk'])


        fig, ax = plt.subplots(figsize=(10, 6))
        cross_tab.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Stacked Bar Plot: Sex vs Risk')
        ax.set_ylabel('Count')
        ax.set_xlabel('Sex')
        plt.xticks(rotation=45)


        st.pyplot(fig)
        with st.expander("**Insights:**"):      
            st.markdown(
            "- This is a stacked bar plot of count of gender vs risk.\n"
            "- This plot shows that the credit risk associated with males is higher by 250 compared to females" 
                " despite the female count being lesser than males.\n"
            "- The overall female count appears to be around 300 while the male count appears to be 700.\n"
            
            )
    
    with tab2:
        st.header("Purpose Count Based on Risk")
        st.write("") 

   
        cross_tab = pd.crosstab(df['Purpose'], df['Risk'])


        fig, ax = plt.subplots(figsize=(10, 6))
        cross_tab.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Stacked Bar Plot: Purpose vs Risk')
        ax.set_ylabel('Count')
        ax.set_xlabel('Purpose')
        plt.xticks(rotation=45)


        st.pyplot(fig)

   
        with st.expander("**Insights:**"):
            st.write("- This is a stacked bar plot of purpose count based on risk.\n"
                    "- Car loans appear to be the most common type of loan and is associated with a good credit risk profile .\n"
                    "- On the contrary loans for  domestic appliances appear to be the rarest.\n"
                    "- Almost all types of loans have good credit risk except for Furniture/equipment \n"
                    "- It's inteersting to note that the credit risk associated with car and furniture is almost the same"  )
    with tab3:
        st.header("Credit Amount Group Based on Risk")
        st.write("")  # Empty line for spacing

       
        df['Credit Amount Group'] = pd.cut(df['Credit amount'], bins=[0, 500, 1000, 5000, 10000, 50000], 
                                        labels=['Low', 'Medium', 'High', 'Very High', 'Extremely High'])


        cross_tab_credit_risk = pd.crosstab(df['Credit Amount Group'], df['Risk'])

      
        fig, ax = plt.subplots(figsize=(10, 6))
        cross_tab_credit_risk.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Credit Amount Group vs Credit Risk')
        ax.set_ylabel('Count')
        ax.set_xlabel('Credit Amount Group')
        plt.xticks(rotation=45)

    
        st.pyplot(fig)

       
        with st.expander("**Insights:**"):
            st.write("- This plot shows the relationship between credit amount groups and the associated risk levels.\n "
                    "- We can observe how different levels of credit amount are distributed across various risk categories.\n"
                    "- We can observe that the credit risk increases with credit amount. This could possibly be due to higher rates and monthly payments to be made")

if selected=="credit amount vs duration":
    st.title("Credit Amount vs Duration")
    st.divider()

    aqi_option = st.selectbox("**Select one of the comparisons below and the corresponding graph will be displayed**",("Lineplot based off Duration", "Histograms of credit and duration"),
        index=None,
        placeholder="Select",
    )
    st.write("")
    
    st.write('**You selected   :**', aqi_option)
    st.divider()
   
    if aqi_option=="Lineplot based off Duration":
        st.subheader("Lineplot")
        
        grped_hr = df.groupby("Duration").mean(numeric_only=True)
        grped_hr_index = grped_hr.index

     
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(grped_hr_index, grped_hr['Credit amount'], marker='o', color='red', linestyle='-')

   
        ax.set_title('Mean Credit Amount with Duration')
        ax.set_xlabel('Duration')
        ax.set_ylabel('Mean Credit Amount')

      
        st.pyplot(fig)
        st.write("The plot shows us that the highest credit amount is around 12k associated with the highest duration of 55 months "
        "wheras the lowest credit amount is 1k with the lowest duration of 5 months")

    elif aqi_option=="Histograms of credit and duration" :
        st.subheader("Histograms of Credit Amount and Duration")

        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5)) 

        df['Credit amount'].hist(bins=20, ax=ax[0], color='skyblue', edgecolor='black')
        ax[0].set_title('Credit Amount Distribution')
        ax[0].set_xlabel('Credit Amount')
        ax[0].set_ylabel('Count')

        df['Duration'].hist(bins=20, ax=ax[1], color='salmon', edgecolor='black')
        ax[1].set_title('Duration Distribution')
        ax[1].set_xlabel('Duration')
        ax[1].set_ylabel('Count')

        plt.suptitle('Histograms of Numerical Features')
        plt.tight_layout()

        st.pyplot(fig)
        st.write("")
        st.write("The first histogram shows us that  most borrrowers have a credit amount of 2k to 2.5k ")
if selected=="age based analysis"  :
    st.title("age based analysis")
    st.divider()
   
    bins = [20, 30, 50, 60, 80]
    labels = ['20-30', '30-50', '50-60', '70-80']


    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
   
    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(x='AgeGroup', hue='Purpose', data=df, ax=ax)

    ax.set_title('Purpose Count Across Age Groups')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Purpose', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

  
    st.pyplot(fig)

    st.write("This plot shows the different purposes for which various age groups take loans."
     "According to the plot people from 30-50 take the most loans. This makes sense as this group is typically in its prime working years ")

    st.title("KDE Plot")
    st.divider()

    fig, ax = plt.subplots(figsize=(8,5))
    for risk_value in df['Risk'].unique():
        sns.kdeplot(
            df[df['Risk'] == risk_value]['Age'], 
            label=f'Risk {risk_value}', 
            ax=ax
        )
    ax.set_title('Age Distribution by Credit Risk')
    ax.set_xlabel('Age')
    ax.set_ylabel('Density')
    ax.legend()

    st.pyplot(fig)
    st.write(" A KDE plot (Kernel Density Estimate plot) shows the probability density of a continuous variable. "
    "The plot shows that  5% of the total population for that risk class 0 belongs to Age 30 while risk class 1 belomgs to age 20.")

if selected=="outlier analysis"  :
    st.title("outlier plots")
    st.divider()
    col1, col2 = st.columns(2)

   
    with col1:
        st.subheader("Scatter Plot: Age vs Credit Amount")
        fig1, ax1 = plt.subplots(figsize=(6,5))
        sns.scatterplot(x=df['Age'], y=df['Credit amount'], hue=df['Risk'], palette='coolwarm', alpha=0.7, ax=ax1)
        ax1.set_title('Age vs Credit Amount (by Risk)')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Credit Amount')
        ax1.legend(title='Risk')
        st.pyplot(fig1)

        with st.expander("Insights"):
            st.markdown(
                "- Visualizes how credit amount varies with age based on risk.\n"
                "- According to the plot most of the values appear to be clustered around ages 20–50 and credit amounts below 10k.\n"
                "- However there are a couple of very high credit loans above 120k which could be classified as potential outliers\n"
                "- The highest credit loan availed is approximately 175k"
            )

    # Second plot: Boxplot
    with col2:
        st.subheader("Boxplot of Credit Amount")
        fig2, ax2 = plt.subplots(figsize=(6,5))
        sns.boxplot(x=df['Credit amount'], ax=ax2)
        ax2.set_title('Boxplot of Credit Amount')
        ax2.set_xlabel('Credit Amount')
        st.pyplot(fig2)

        with st.expander("Insights"):
            st.markdown(
                "- Boxplot that analyses Credit Amount to check for outliers.\n"
            )

    
if selected=="Types of housing and savings analysis"  :
    
    st.divider()
    tab1, tab2= st.tabs(["Types of housing","Saving and Checking Account Analysis"])
    with tab1:
        st.header("Housing Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Housing Type Distribution (Pie Chart)")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            df['Housing'].value_counts().plot.pie(
                autopct='%1.1f%%', 
                startangle=90, 
                ax=ax1
            )
            ax1.set_ylabel('')
            ax1.set_title('Housing Type Distribution')
            st.pyplot(fig1)

            with st.expander(" Pie Chart Insights"):
                st.write("This pie chart shows how housing types are distributed across the dataset. "
                        "It gives a quick visual breakdown of which housing option is most common."
                        "It shows that 71.3% population who take loans have their own housing")

        with col2:
            st.subheader("Housing Type Count (Bar Chart)")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            sns.countplot(x='Housing', data=df, ax=ax2)
            ax2.set_title('Count of Each Housing Type')
            st.pyplot(fig2)

            with st.expander(" Bar Chart Insights"):
                st.write("This bar chart helps compare the number of applicants in each housing category. "
                        "It is useful for identifying the most and least common housing types.")

    with tab2:
        st.header("Account Types Analysis")

        categorical_cols = ['Saving accounts', 'Checking account']
        col1, col2 = st.columns(2)

        for i, col in enumerate(categorical_cols):
            with [col1, col2][i]:
                st.subheader(f"{col} vs Checking Account")

                fig, ax = plt.subplots(figsize=(7, 4))
                sns.countplot(x=col, data=df, hue='Checking account', ax=ax)
                ax.set_title(f'Count Plot of {col}')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)

                with st.expander(f" Insights on {col}"):
                    st.write(f"This plot shows the distribution of **{col}** categorized by different **Checking account** values. "
                            f"It helps in understanding the relationship between {col.lower()} and checking account status.")


if selected =="ml model analysis":
    st.title("Random Forest Classifier")
    st.divider()
      
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.metrics import confusion_matrix
    global preprocessor,model
    st.header("Most influential features")
    st.write("The most influential features from analsying the relation between risk and different features appears to be Credit Amount,Purpose, Types of Housing and Age")
  
    # features and target variable
    X = df.drop('Risk', axis=1)  # Features
    y = df['Risk']  # Target

    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    numerical_cols = ['Age', 'Credit amount', 'Duration']

    # preprocess data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols), 
            ('cat', OneHotEncoder(drop='first'), categorical_cols) 
        ])


    # fit and transform data
    X_processed = preprocessor.fit_transform(X)
  
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    
    model = RandomForestClassifier(n_estimators=100,class_weight='balanced', random_state=42)

    #train data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

 
    st.subheader("Classification Report")
    st.dataframe(report_df.style.format(precision=2))

   
    metrics = {
        'Accuracy': [accuracy_score(y_test, y_pred)],
        'Precision': [precision_score(y_test, y_pred)],
        'Recall': [recall_score(y_test, y_pred)],
        'F1-Score': [f1_score(y_test, y_pred)],
    }
    metrics_df = pd.DataFrame(metrics)

    st.subheader(" Overall Model Metrics")
    st.table(metrics_df.style.format(precision=2))

    cm = confusion_matrix(y_test, y_pred)


    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good Credit', 'Bad Credit'], 
                yticklabels=['Good Credit', 'Bad Credit'],
                ax=ax_cm)
    ax_cm.set_title('Confusion Matrix')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)
    st.write("This confusion matrix summarizes the performance of the Random Forest Classifier in distinguishing between good and bad credit."
    "The model correctly predicted 108 cases of good credit and 79 cases of bad credit."
    "It misclassified 10 good credit cases as bad (false positives) and 3 bad credit cases as good (false negatives).")




    
    # Step 1: Define the categorical and numerical columns
    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    numerical_cols = ['Age', 'Credit amount', 'Duration']



    # Step 2: Create Streamlit inputs for dynamic user input
    st.title("Credit Risk Prediction - User Input")


    age = st.number_input('Age', min_value=18, max_value=100
    )
    sex = st.selectbox('Sex', ['male', 'female'])
    job = st.selectbox('Job', [0, 1, 2, 3])  # Numeric encoded values for Job
    housing = st.selectbox('Housing', ['own', 'free', 'for rent'])
    saving_accounts = st.selectbox('Saving accounts', ['little', 'rich', 'moderate', 'poor'])
    checking_account = st.selectbox('Checking account', ['little', 'rich', 'moderate', 'poor'])
    credit_amount = st.number_input('Credit amount', min_value=0, max_value=1000000, value=10000)
    duration = st.number_input('Duration', min_value=1, max_value=72, value=12)
    purpose = st.selectbox('Purpose', ['car', 'education', 'business', 'others'])

    # Step 3: Collect inputs into a dictionary
    val_dict = {
        'Age': age,
        'Sex': sex,
        'Job': job,  
        'Housing': housing,
        'Saving accounts': saving_accounts,
        'Checking account': checking_account,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose
    }

    val_list = list(val_dict.values())
    val_list = [val_list]  

    columns = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
    val_df = pd.DataFrame(val_list, columns=columns)

    if st.button("Calculate Credit Risk"):
        # Step 5: Use the preprocessor to transform the new data
        val_processed = preprocessor.transform(val_df)

        # Step 6: Make prediction
        test_pred = model.predict(val_processed)

        # Step 7: Show result
        if test_pred[0] == 1:
            st.success("Good Credit Score!")
        else:
            st.error("Bad Credit Score!")

    

import streamlit as st
import requests
from requests.exceptions import HTTPError
import json

url = "https://fastapi-openclassrooms.herokuapp.com/predict"

st.set_page_config(
    page_title="Prêt à dépenser - Un algorithme de classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("""
Cette app calculer la probabilité qu’un client rembourse son crédit.
""")

#Former la barre latérale
st.sidebar.header("Données personnelles de l'utilisateur")
st.subheader("Paramètres d'entrée utilisateur")


def user_input_features():
    
    CODE_GENDER= st.sidebar.selectbox('Sexe', ['F','M'])
    DAYS_BIRTH = st.sidebar.number_input('Âge du client en jours au moment de la demande', -30000, -6570)


    NAME_FAMILY_STATUS = st.sidebar.selectbox("Situation familiale du client", ['Single / not married', 'Married', 'Civil marriage', 'Widow', 'Separated'])
    CNT_CHILDREN = st.sidebar.number_input("Nombre d'enfants", 0, 19)
    CNT_FAM_MEMBERS = st.sidebar.number_input('Combien de membres de la famille le client a-t-il', 0,22)
    NAME_TYPE_SUITE = st.sidebar.selectbox("Qui accompagnait le client lors de sa demande de prêt?", ['Unaccompanied', 'Family', 'Spouse, partner', 'Children',
                                           'Other_A', 'Other_B', 'Group of people'])


    NAME_HOUSING_TYPE = st.sidebar.selectbox('Quelle est la situation de logement du client', ['House / apartment', 'Rented apartment', 'With parents',
                                                                                               'Municipal apartment', 'Office apartment', 'Co-op apartment'])
    FLAG_OWN_REALTY = st.sidebar.selectbox("Signaler si le client possède une maison ou un appartement (Y=OUI, N=NON)",['Y','N'])
    FLAG_PHONE = st.sidebar.selectbox("Le client a-t-il fourni un téléphone résidentiel (1=OUI, 0=NON)", [1,0])
    FLAG_CONT_MOBILE = st.sidebar.selectbox("Le téléphone portable était-il joignable (1=OUI, 0=NON)", [1,0])
    DAYS_LAST_PHONE_CHANGE = st.number_input("Combien de jours avant l'application le client a-t-il changé de téléphone", -5000, 0)
    FLAG_EMAIL  = st.sidebar.selectbox("Le client a-t-il fourni un e-mail (1=OUI, 0=NON)", [1,0])
    FLAG_OWN_CAR = st.sidebar.selectbox("Signaler si le client possède une voiture (Y=OUI, N=NON)",['Y','N'])

    NAME_EDUCATION_TYPE = st.sidebar.selectbox("Niveau de scolarité le plus élevé atteint par le client", ['Secondary / secondary special', 'Higher education',
                                                                                                           'Incomplete higher', 'Lower secondary', 'Academic degree'])
    OCCUPATION_TYPE = st.sidebar.selectbox("Quel type d'occupation le client a-t-il", ['Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers', 'Sales staff',
                                                                                       'Cleaning staff', 'Cooking staff','Private service staff', 'Medicine staff',
                                                                                       'Security staff', 'High skill tech staff', 'Waiters/barmen staff',
                                                                                       'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff', 'HR staff'])
    NAME_INCOME_TYPE = st.sidebar.selectbox("Type de revenu des clients", ['Working', 'State servant', 'Commercial associate', 'Pensioner','Unemployed', 'Student',
                                                                           'Businessman', 'Maternity leave'])
    NAME_CONTRACT_TYPE = st.sidebar.selectbox('Identification si le prêt est en espèces ou renouvelable', ['Cash loans', 'Revolving loans'])
    DAYS_EMPLOYED = st.sidebar.number_input("Combien de jours avant la demande la personne a commencé l'emploi actuel?", -18000, 0)
    FLAG_WORK_PHONE  = st.sidebar.selectbox("Le client a-t-il fourni un téléphone portable fonctionnel", [1,0])
    FLAG_EMP_PHONE  = st.sidebar.selectbox("Le client a-t-il fourni un téléphone fixe fonctionnel (1=OUI, 0=NON)", [1,0])

    LIVE_CITY_NOT_WORK_CITY = st.sidebar.selectbox("Signaler si l'adresse de contact du client ne correspond pas à l'adresse professionnelle 1=différent, 0=identique, au niveau de la ville)",
                                                       [1,0])
    LIVE_REGION_NOT_WORK_REGION = st.sidebar.selectbox("Signaler si l'adresse de contact du client ne correspond pas à l'adresse professionnelle (1=différent, 0=identique, au niveau de la ville)",
                                                       [1,0])
    REG_CITY_NOT_WORK_CITY = st.sidebar.selectbox("Signaler si l'adresse permanente du client ne correspond pas à l'adresse professionnelle (1=différent, 0=identique, au niveau de la ville)",
                                                  [1,0])
    REG_CITY_NOT_LIVE_CITY = st.sidebar.selectbox("Signaler si l'adresse permanente du client ne correspond pas à l'adresse de contact (1=différent, 0=identique, au niveau de la ville)",
                                                  [1,0])
                     
    REG_REGION_NOT_WORK_REGION = st.sidebar.selectbox("Signaler si l'adresse de contact du client ne correspond pas à l'adresse professionnelle (1=différent, 0=identique, au niveau de la région)",
                                                      [1,0])
        
    DAYS_ID_PUBLISH=st.sidebar.number_input("Combien de jours avant le dépôt de la demande le client a-t-il changé la pièce d'identité avec laquelle il a demandé le service?",
                                        -8000, 0)
    
    DAYS_REGISTRATION = st.sidebar.number_input("Combien de jours avant la demande le client a-t-il modifié son inscription?", -25000, 0)
    
    
    AMT_GOODS_PRICE = st.sidebar.number_input("Pour les prêts à la consommation c'est le prix des biens pour lesquels le prêt est accordé", 10000, 5000000)
    AMT_ANNUITY = st.sidebar.number_input("Rente de prêt", 1000, 300000)
    AMT_CREDIT = st.sidebar.number_input('Montant du crédit du prêt', 30000, 4500000)
    AMT_INCOME_TOTAL = st.sidebar.number_input('Revenu du client', 10000, 17000000)
    
    
    AMT_REQ_CREDIT_BUREAU_YEAR = st.number_input('Nombre de demandes de renseignements au bureau de crédit concernant le client', 0, 30)
    HOUR_APPR_PROCESS_START = st.number_input('À quelle heure environ le client a-t-il demandé le prêt', 0, 24)
    WEEKDAY_APPR_PROCESS_START = st.selectbox('Quel jour de la semaine le client a-t-il demandé le prêt', ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY',
                                                                                                           'SATURDAY', 'SUNDAY'])

    REGION_RATING_CLIENT_W_CITY  = st.selectbox("Notre évaluation de la région où vit le client en tenant compte de la ville (1,2,3)", [1,2,3])
    REGION_RATING_CLIENT = st.selectbox("Notre évaluation de la région où vit le client (1,2,3)", [1,2,3])
    REGION_POPULATION_RELATIVE = st.number_input("Population normalisée de la région dans laquelle vit le client (un nombre plus élevé signifie que le client vit dans une région plus populaire)",
                                                 0.0, 0.10)

    FLAG_DOCUMENT_21 = st.selectbox('Le client a-t-il fourni le document 21 (1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_20 = st.selectbox('Le client a-t-il fourni le document 20(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_19 = st.selectbox('Le client a-t-il fourni le document 19(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_18 = st.selectbox('Le client a-t-il fourni le document 18(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_17 = st.selectbox('Le client a-t-il fourni le document 17(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_16 = st.selectbox('Le client a-t-il fourni le document 16(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_15 = st.selectbox('Le client a-t-il fourni le document 15(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_14 = st.selectbox('Le client a-t-il fourni le document 14(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_13 = st.selectbox('Le client a-t-il fourni le document 13(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_11 = st.selectbox('Le client a-t-il fourni le document 11(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_9 = st.selectbox('Le client a-t-il fourni le document 9(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_8 = st.selectbox('Le client a-t-il fourni le document 8(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_7 = st.selectbox('Le client a-t-il fourni le document 7(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_6 = st.selectbox('Le client a-t-il fourni le document 6(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_5 = st.selectbox('Le client a-t-il fourni le document 5(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_3 = st.selectbox('Le client a-t-il fourni le document 3(1=OUI, 0=NON)', [1,0])
    FLAG_DOCUMENT_2 = st.selectbox('Le client a-t-il fourni le document 2(1=OUI, 0=NON)', [1,0])
    
    OBS_30_CNT_SOCIAL_CIRCLE = st.selectbox("Combien d'observations de l'environnement social du client avec un défaut observable de 30 DPD (jours de retard)", [0,1,2,3,4,5])
    OBS_60_CNT_SOCIAL_CIRCLE = st.selectbox("Combien d'observations de l'environnement social du client avec un défaut observable de 60 DPD (jours de retard)", [0,1,2,3,4,5])

    EXT_SOURCE_2 = st.number_input("Score normalisé à partir d'une source de données externe (2)", 0.0, 1.0)
    EXT_SOURCE_3 = st.number_input("Score normalisé à partir d'une source de données externe (3)", 0.0, 1.0)


    return {'CODE_GENDER': CODE_GENDER,
            'DAYS_BIRTH': DAYS_BIRTH,
            'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS,
            'CNT_CHILDREN': CNT_CHILDREN,
            'CNT_FAM_MEMBERS': CNT_FAM_MEMBERS,
            'NAME_TYPE_SUITE' : NAME_TYPE_SUITE,
            'NAME_HOUSING_TYPE': NAME_HOUSING_TYPE,
            'FLAG_OWN_REALTY' : FLAG_OWN_REALTY,
            'FLAG_PHONE': FLAG_PHONE,
            'FLAG_CONT_MOBILE' : FLAG_CONT_MOBILE,
            'DAYS_LAST_PHONE_CHANGE': DAYS_LAST_PHONE_CHANGE,
            'FLAG_EMAIL' : FLAG_EMAIL,
            'FLAG_OWN_CAR' : FLAG_OWN_CAR,
            'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
            'OCCUPATION_TYPE': OCCUPATION_TYPE,
            'NAME_INCOME_TYPE' : NAME_INCOME_TYPE,
            'NAME_CONTRACT_TYPE' : NAME_CONTRACT_TYPE,
            'DAYS_EMPLOYED' : DAYS_EMPLOYED,
            'FLAG_WORK_PHONE' : FLAG_WORK_PHONE,
            'FLAG_EMP_PHONE' : FLAG_EMP_PHONE,
            'LIVE_CITY_NOT_WORK_CITY': LIVE_CITY_NOT_WORK_CITY,
            'LIVE_REGION_NOT_WORK_REGION' : LIVE_REGION_NOT_WORK_REGION,
            'REG_CITY_NOT_WORK_CITY' : REG_CITY_NOT_WORK_CITY,
            'REG_CITY_NOT_LIVE_CITY': REG_CITY_NOT_LIVE_CITY,
            'REG_REGION_NOT_WORK_REGION' : REG_REGION_NOT_WORK_REGION,
            'DAYS_ID_PUBLISH' : DAYS_ID_PUBLISH,
            'DAYS_REGISTRATION': DAYS_REGISTRATION,
            
            

            'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
            'AMT_ANNUITY': AMT_ANNUITY,
            'AMT_CREDIT': AMT_CREDIT,
            'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
            'AMT_REQ_CREDIT_BUREAU_YEAR': AMT_REQ_CREDIT_BUREAU_YEAR,
            'HOUR_APPR_PROCESS_START': HOUR_APPR_PROCESS_START,
            'WEEKDAY_APPR_PROCESS_START' : WEEKDAY_APPR_PROCESS_START,
            'REGION_RATING_CLIENT_W_CITY':REGION_RATING_CLIENT_W_CITY,
            'REGION_RATING_CLIENT' : REGION_RATING_CLIENT,
            'REGION_POPULATION_RELATIVE' : REGION_POPULATION_RELATIVE,
            'FLAG_DOCUMENT_21': FLAG_DOCUMENT_21,
            'FLAG_DOCUMENT_20' : FLAG_DOCUMENT_20,
            'FLAG_DOCUMENT_19' : FLAG_DOCUMENT_19,
            'FLAG_DOCUMENT_18' : FLAG_DOCUMENT_18,
            'FLAG_DOCUMENT_17' : FLAG_DOCUMENT_17,
            'FLAG_DOCUMENT_16' : FLAG_DOCUMENT_16,
            'FLAG_DOCUMENT_15' : FLAG_DOCUMENT_15,
            'FLAG_DOCUMENT_14' : FLAG_DOCUMENT_14,
            'FLAG_DOCUMENT_13' : FLAG_DOCUMENT_13,
            'FLAG_DOCUMENT_11' : FLAG_DOCUMENT_11,
            'FLAG_DOCUMENT_9' : FLAG_DOCUMENT_9,
            'FLAG_DOCUMENT_8' :  FLAG_DOCUMENT_8,
            'FLAG_DOCUMENT_7' : FLAG_DOCUMENT_7,
            'FLAG_DOCUMENT_6' : FLAG_DOCUMENT_6,
            'FLAG_DOCUMENT_5' : FLAG_DOCUMENT_5,
            'FLAG_DOCUMENT_3' : FLAG_DOCUMENT_3,
            'FLAG_DOCUMENT_2' : FLAG_DOCUMENT_2, 
            'OBS_30_CNT_SOCIAL_CIRCLE': OBS_30_CNT_SOCIAL_CIRCLE,
            'OBS_60_CNT_SOCIAL_CIRCLE': OBS_60_CNT_SOCIAL_CIRCLE,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'EXT_SOURCE_3': EXT_SOURCE_3,
            
                }

data_in = user_input_features()
st.write(data_in)

if st.button('Predict'):

    try:
        response = requests.post(url = url, data = json.dumps(data_in))
        st.text(f"API status code: {response.status_code}")
        response.raise_for_status()
    except HTTPError as http_err:
        st.text(f"HTTP error occurred: {http_err}")
        st.text(f"{response.text}")
    except Exception as err:
        st.text(f"error occurred: {err}")
    else:
        st.text(f"API answered: {response.json()}")

import pandas as pd
from preprocessing import preprocess
from delay_regression import train_delay_regression , evaluate_regression
from delay_classification import train_delay_classification , evaluate_classification

try :

    df = pd.read_csv(r"dataset/flights.csv")

    preprocessed_df = preprocess(df)


    while True :

        print("CHOICE                  ACTION")
        print("  1     :  Predict arrival delay (minutes)")
        print("  2     :  Predict delay status (delayed / not delayed)")
        print("  3     :  EXIT")
        print()

        choice = int(input("Enter Your Choice : "))
        print()

        match choice :

            case 1 :
                model , X_test , Y_test = train_delay_regression(preprocessed_df)
                Y_pred , metric = evaluate_regression(model , X_test , Y_test)
                print(Y_pred[:20])
                print()
                print(metric)
                print()

            case 2 :
                model , X_test , Y_test = train_delay_classification(preprocessed_df)
                Y_pred , metric = evaluate_classification(model , X_test , Y_test)
                print(Y_pred[:20])
                print()
                print(metric)
                print()

            case 3 :
                print("SUCCESFULLY EXITED")
                break

            case _ :
                print("INVALID OPTION")
                print()

except ValueError :
    print("INVALID OPTION (Enter Valid Number)")
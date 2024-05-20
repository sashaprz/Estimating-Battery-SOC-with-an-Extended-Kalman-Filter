import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from lmfit import Model

# Function to fit the curve
def equation(z, K0, K1, K2, K3, K4):
    return K0 - K1/z - K2*z + K3*np.log(z) + K4*np.log(1-z)

# Function to authenticate and get data from Google Sheets
def get_data_from_google_sheets(file_name, sheet_name, range_name):
    try:
        # Define the scope
        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

        # Add your own credentials file
        creds = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\Sasha\Downloads\ocv-soc-curve-data-k-function-24d2b9e31aa9.json", scope)
        client = gspread.authorize(creds)

        # Open the Google Sheet by name
        sheet = client.open(file_name).worksheet(sheet_name)  # Access the specific sheet within the spreadsheet
        data = sheet.get(range_name)
        
        if not data or len(data) <= 1:
            print("No data found in the specified range or only header present.")
            return None

        # Convert the data to numpy arrays, skipping the first row (header)
        data = np.array(data[1:], dtype=np.float64)  # Skip header and convert to float
        x_data = data[:, 0]
        y_data = data[:, 1]

        return x_data, y_data
    except gspread.exceptions.APIError as e:
        print(f"API error: {e}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

# Main function
def main():
    # Google Sheets info
    FILE_NAME = 'OCV_SOC curve values'  # Replace with the name of your Google Sheet file
    SHEET_NAME = 'Sheet1'  # Replace with the name of the specific sheet within the Google Sheet file
    RANGE_NAME = 'A:B'  # Adjust the range if necessary

    # Get data from Google Sheets
    result = get_data_from_google_sheets(FILE_NAME, SHEET_NAME, RANGE_NAME)
    if result is None:
        print("Failed to retrieve data.")
        return
    
    x_data, y_data = result

    initial_guess = (4.0, 1.0, 1.0, 1.0, 2.0)  # Replace with your initial guesses for K0, K1, K2, K3, K4

    # Curve fitting with initial parameter guesses
    popt, pcov = curve_fit(equation, x_data, y_data, p0=initial_guess, maxfev=50000)

    # Curve fitting
    #popt, pcov = curve_fit(equation, x_data, y_data, maxfev=50000)

    # Print the fitted equation
    K0, K1, K2, K3, K4 = popt
    print(f"Fitted equation: ocv = {K0} - {K1}/z - {K2}*z + {K3}*np.log(z) + {K4}*np.log(1-z)")

    # Plotting
    plt.scatter(x_data, y_data, label='Original Data')
    z_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = equation(z_fit, *popt)
    plt.plot(z_fit, y_fit, color='red', label='Fitted Curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

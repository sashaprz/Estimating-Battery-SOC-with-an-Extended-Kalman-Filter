#use this code to do regression on your SOC-OCV data and determine the values for K1, K2, K3, and K4, which are needed in the C matrix and in the OCV function. 

#import necessary libraries
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from scipy.optimize import curve_fit

#define the form that you are regressing the data into. I.e., this equation form where you are solving for the K coefficients
def model(z, K0, K1, K2, K3, K4):
    return K0 - K1/z - K2*z + K3*np.log(z) + K4*np.log(1-z)

# Google Sheets API setup
def get_data_from_sheet(sheet_url, sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(r"xxxx", scope) #add the path to your json file for the API key
    client = gspread.authorize(creds)
    
    sheet = client.open_by_url(sheet_url).worksheet(sheet_name)
    data = sheet.get_all_values()
    
    # Convert data to numpy array and extract x (z) and y values
    data = np.array(data[2:], dtype=float)  # assuming first row is header
    z_initial = data[:, 0] 
    y = data[:, 1]

    #dividing z by 100 to convert to percantage values to avoid log error
    z = z_initial / 100
    
    return z, y

# Fit the model
def fit_model(z, y):
    # Initial guess for the coefficients (can be adjusted)
    initial_guess = [1, 1, 1, 1, 1]
    
    try:
        # Use curve_fit to fit the model
        popt, pcov = curve_fit(model, z, y, p0=initial_guess)
        return popt, pcov
    #error handling
    except RuntimeError as e:
        print(f"Fit could not be performed: {e}")
        return None, None

# Main function
def main():
    # Replace with your Google Sheet URL and sheet name
    sheet_url = "https://docs.google.com/spreadsheets/d/1AcGhJumi5lQYReXOr7019HTJ4Q93r5n42zfjYVU2fgY/edit#gid=0"
    sheet_name = "Sheet1"

    #extract data from sheet
    z, y = get_data_from_sheet(sheet_url, sheet_name)
    
    # Check for NaN or Inf values and remove them
    valid_indices = ~(np.isnan(z) | np.isnan(y) | np.isinf(z) | np.isinf(y))
    z = z[valid_indices]
    y = y[valid_indices]
    
    # Fit the model and get the coefficients - call the function defined above
    coefficients, covariance = fit_model(z, y)

    #print results
    print("Fitted coefficients:")
    print(f"K0: {coefficients[0]}")
    print(f"K1: {coefficients[1]}")
    print(f"K2: {coefficients[2]}")
    print(f"K3: {coefficients[3]}")
    print(f"K4: {coefficients[4]}")

if __name__ == "__main__":
    main()

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from scipy.optimize import curve_fit

# Function to define the model
def model(z, K0, K1, K2, K3, K4):
    return K0 - K1/z - K2*z + K3*np.log(z) + K4*np.log(1-z)

# Google Sheets setup
def get_data_from_sheet(sheet_url, sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\Sasha\Downloads\ocv-soc-curve-data-k-function-24d2b9e31aa9.json", scope)
    client = gspread.authorize(creds)
    
    sheet = client.open_by_url(sheet_url).worksheet(sheet_name)
    data = sheet.get_all_values()
    
    # Convert data to numpy array and extract x (z) and y values
    data = np.array(data[2:], dtype=float)  # assuming first row is header
    z_initial = data[:, 0] 
    y = data[:, 1]

    #diviing z by 100 to convert to percantage values to avoid log error
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
    except RuntimeError as e:
        print(f"Fit could not be performed: {e}")
        return None, None

# Main function
def main():
    # Replace with your actual Google Sheet URL and sheet name
    sheet_url = "https://docs.google.com/spreadsheets/d/1AcGhJumi5lQYReXOr7019HTJ4Q93r5n42zfjYVU2fgY/edit#gid=0"
    sheet_name = "Sheet1"
    
    z, y = get_data_from_sheet(sheet_url, sheet_name)
    
    # Check for NaN or Inf values and remove them
    valid_indices = ~(np.isnan(z) | np.isnan(y) | np.isinf(z) | np.isinf(y))
    z = z[valid_indices]
    y = y[valid_indices]
    
    # Fit the model and get the coefficients
    coefficients, covariance = fit_model(z, y)
    
    print("Fitted coefficients:")
    print(f"K0: {coefficients[0]}")
    print(f"K1: {coefficients[1]}")
    print(f"K2: {coefficients[2]}")
    print(f"K3: {coefficients[3]}")
    print(f"K4: {coefficients[4]}")

if __name__ == "__main__":
    main()
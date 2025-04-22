#use this to extract the data from your OCV discharge of 100% to 0% at low, constant, current, and integrate the curve. This allows you to generate an SOC-OCV curve, which will then be exported to a new google sheet. 

#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Function to extract data from the first Google Sheet
def extract_data_from_first_sheet(sheet_name, credentials_file):
    scope = ["//link to your spreadsheet"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(credentials)
    sheet = client.open(sheet_name).sheet1
    data = sheet.get_all_values()
    return np.array(data)

# Function to write data to a new Google Sheet
def write_to_new_sheet(data, sheet_name, credentials_file):
    scope = ["//link to your spreadsheet"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(credentials)
    new_sheet = client.create(sheet_name)
    worksheet = new_sheet.sheet1
    for row in data:
        worksheet.append_row(row)

# Assuming the name of the first Google Sheet and the credentials file
first_sheet_name = 'xxx' #name of your first sheet (where you get the raw data from)
credentials_file = r'xxx' #path to the json file with the API key for the first sheet

# Step 1: Extract data from the first Google Sheet
ocv_data = extract_data_from_first_sheet(first_sheet_name, credentials_file)

# Extract OCV values from the extracted data - i.e. formatting
ocv_values = np.array(ocv_data)[:, 1].astype(float)

# Integrate to assign SOC values
soc_values = np.linspace(100, 0, len(ocv_values))

# Prepare data for writing to a new sheet
data_to_write = [['SOC (%)', 'OCV']]
for soc, ocv in zip(soc_values, ocv_values):
    data_to_write.append([soc, ocv])

# Assuming the name of the new Google Sheet
new_sheet_name = 'xxx' #your second sheet's name
new_credentials_file = r'xxx' #path to the json file with the API key for your second sheet

# Step 2: Write interpolated data to a new Google Sheet
write_to_new_sheet(data_to_write, new_sheet_name, new_credentials_file)

# Step 3: Plot SOC vs OCV
plt.plot(soc_values, ocv_values)
plt.xlabel('SOC (%)')
plt.ylabel('OCV')
plt.title('SOC vs OCV')
plt.grid(True)
plt.show()

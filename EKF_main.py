#LIBRARIES
import numpy as np
import sympy
import sympy as sp
from sympy.stats import Normal
from sympy import And
from sympy import Eq
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import inspect
from time import sleep
from decimal import Decimal, getcontext

#setting up google sheets API
sheet_name = 'xxx' #name of sheet with your original collected data during the OCV discharge
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(r"xxx, scope) #path to the json file of your API key

gc = gspread.authorize(credentials)
client = gspread.authorize(credentials)
sheet = client.open(sheet_name).sheet1

#extracting data + formattin
data = sheet.get_all_records()
np_data = np.array(data)
columns = list(data[0].keys())
matrix_data = [list(row.values()) for row in data]

# Convert the 2D list to a NumPy array
np_data = np.array(matrix_data)

# Assuming the sheet columns are interval number, voltage, and current
interval_numbers = np_data[:, 0].astype(int)  # First column
voltage = np_data[:, 1].astype(float)         # Second column
current = np_data[:, 2].astype(float)         # Third column

# Create a structured data dictionary to match the original format
structured_data = {
    'interval': interval_numbers,
    'voltage': voltage,
    'current': current
}

current_values = current
voltage_values = voltage

first_current = current_values[0]
first_voltage = voltage_values[0]

#FUNCTIONS

#these need to be calculated using the pulse discharge test and the graph (see readme)
R0 = 0.0121
R1 = 0.0155
C1 = 166.8685/0.0155
Q = 0.800*3600 # Convert [amp-hour] to [amp-seconds]. also, the first value is capacity of your battery
eta = 1
deltaT = 1 

#computing the A B and D matrices
A = np.array([[1,0],[0, np.exp( -deltaT/(R1*C1) ) ]])
B = np.array([ [-eta*deltaT/Q], [ R1 * (1-np.exp(-deltaT/(R1*C1) ) ) ] ])
D = np.array([[-R0]])

#f function - uses what it knows about current, efficiency, etc, to predict how much SOC should decrease by
def f_function(x, u):
    x_new = A@x + B*u
    return x_new

#relates SOC to OCV
def OCV(z): 
    #replace with the coefficients you calculated in the generate_SOC_OCV_curve file
    K0 = 6.772893107950994
    K1 = - 0.04243404395791764
    K2 = 4.215886846254147
    K3 = 2.088959891840822
    K4 = - 0.15208255568679976
    
    ocv = K0 - K1/z - K2*z + K3*np.log(z) + K4*np.log(1-z)
    return ocv 

#relates OCV to terminal voltage
def g(x,u): 
    y = OCV(x[0,0]) - x[1,0] + D*u
    return y

#C matrix
def C(x):
    K0 = 6.772893107950994
    K1 = - 0.04243404395791764
    K2 = 4.215886846254147
    K3 = 2.088959891840822
    K4 = - 0.15208255568679976

    c = np.array([[K1/x[0,0]**2 - K2 + K3/x[0,0] - K4/(1-x[0,0]), -1]])
    return c

#to make a pretty graph after :) 
def plot_SOC(estimations, intervals): 

    estimations = np.array(estimations)
    intervals = np.array(intervals)

    # Ensure estimations is 1D for plotting
    if estimations.ndim != 1:
        raise ValueError("Estimations should be a 1D array of values.")
    
    # Ensure intervals is 1D for plotting
    if intervals.ndim != 1:
        raise ValueError("Intervals should be a 1D array of values.")

    #creates a visual representation of the battery discharging. 
    plt.figure(figsize=(8, 6))

    plt.plot(intervals, estimations, marker='o', markersize=3, linestyle='-', label='SOC predictions')

    plt.ylabel('State of Charge Prediction(%)')
    plt.xlabel('Interval Number')
    plt.title('SOC predictions')
    plt.grid(True)
    plt.legend()
    plt.show()
    return


#EKF equations
def main ():
    #VARIABLES
    x_k_plus = 0.0
    x_k_minus =  0.0
    x_k_minus_1 = 1.00
    
    #will need to tune to improve accuracy - error terms
    Sigma_w = np.diag([(1e-4)**2, (1e-2)**2])
    Sigma_v = (1e-1)**2

    y = voltage
    u = current

    #error matrices and terms
    sigma_k_minus_1 = np.eye(2, 2) #initalizing error covariance matrix as equivalent to identify matrix
    x_k_minus_1 =  1.00 #final guess for previous timestep (my data starts as fully charged, therefore 100% SOC)
    vk_scalar = 0.0 #this is an error term but I took it out for simplicity. 
    
    getcontext().prec = 10000

    #measured values (extracted from google sheet later)
    i_measured = 0.0

    #adding returned values to lists so data can be plotted afterwards
    estimations = []
    intervals = []

    time = np.arange(len(current))*deltaT
    N_time = len(time)

    xhat = np.zeros((2,N_time)) # state estimate. note states are two-dimensional vectors.

    sigma_k_minus = np.eye(2,2) # initial guess for covariance martix of states.
    x0_guess = np.array([1,0]) 
    xhat[:,0] = x0_guess # store your initial guess 

    
    for k in range(1, 200): #you can change to have it run longer
        #this iterates through the actual extended kalman filter equations, and returns the error term as well as the predicted SOC value for that timestep (ie, iteration number)
        xhat_prev = xhat[:,k-1][:,None]    

        #extract current and voltage value from google sheet to use in calculations
        row_index = k + 1
        i_measured = float(sheet.cell(row_index, 3).value)
        V_measured = float(sheet.cell(row_index, 2).value)

        #calculate x_k_minus_1 (initial guess) with state space model
        xhat_minus = f_function(xhat_prev, i_measured)
        
        #calculating initial error matrix
        #the reason why it's sigma k minus twice is because im computing the one for the current timestep using the value that was computed for the last timestep. it's basically 2 variables, just simpler because I'm referring to them as one and reassigning. 
        sigma_k_minus = A @ (sigma_k_minus) @ A.T + Sigma_w 

        Chat = C(xhat_minus)

        #calculate kalman gain
        L = sigma_k_minus @ Chat.T @ np.linalg.inv(Chat @ sigma_k_minus @ Chat.T + Sigma_v)

        #calculate estimated voltage measuerement
        y_est = g(xhat_minus, u[k]) #this might need to be k-1 

        #final prediction, based on delta between actual and predicted voltage measurement as well as the kalman gain
        xhat_plus = xhat_minus + L*(voltage[k] - y_est)

        #extracting SOC estimation from the matrix
        xhat[:,k] = xhat_plus[:,0]
        x_k_plus = xhat_plus[0, 0]
        
        #calculating covariance matrix
        #again, it's technically sigma k plus, but using same variable for simplicity
        sigma_k_minus = (np.eye(2,2) - L@Chat)@sigma_k_minus

        #printing results
        print("interval: ", k)
        print("x_k_plus: ", x_k_plus)
        estimations.append(x_k_plus)
        intervals.append(k)

        #adding delay to account for the API limit of 300 requests/minute 
        sleep(2)

    #once iterations finish, show graph
    plot_SOC(estimations, intervals)

    return estimations, intervals
    
main()

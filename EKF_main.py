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
import time

#setting up google sheets API
sheet_name = 'OCV_data_GOOD'
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\Sasha\OneDrive\C_Projects\Simulator\psyched-upgrade-416312-3b5d91fc6159.json", scope)

gc = gspread.authorize(credentials)
client = gspread.authorize(credentials)
sheet = client.open(sheet_name).sheet1

#FUNCTIONS

def f_function(x_k_minus_1, i_measured):
    #the f function represents part of the state space model: the physics-based model of my battery

    n = 0.90 #representing eta, the efficiency factor. LFP batteries have an efficiency of 90-95%
    Q = 0.800 #capacity in Amp-hours
    dt = 1 #~1 second delta t, size of timestep

    # guess for x_k_minus_1 based on state space model
    result = (x_k_minus_1 - ((n * dt / Q ) * i_measured))

    x_k_minus = result 

    return x_k_minus

def OCV_function(x_k_minus):
    #the OCV function is also a part of my state space model, it is used in calculating the g function.  I got this function from creating an SOC-OCV curve (see Arduino code (generating constant current) for more details!)

    function = 0.2218 + 0.2331 * x_k_minus - 0.0062 * x_k_minus**2 + 0.0001 * x_k_minus**3 - 0.0000 * x_k_minus**4

    return function

#Observation Model
#I am using a RC model to represent my battery. This is a type of equivalent circuit model (ECM). These values were measured from my battery (see Arduino code (discharge pulse) for more details on how.) 
def calculate_R1_R0():
    #since R1 and R0 are interconnected (you require one to find the other), they must be solved together. Therefore must use guess and check
    #max_iterations = 1
    #tolerance = 1e-6

    #intial value for R0, based off measurement but doesn't account for variations while SOC decreases. actual measurement was 0.385 but i increased so that it was a more even split between two resistors
    #R0 = 2.385 
    #V_measured = float(V_measured)

    #for num in range(max_iterations):
    #    R1_new = (OCV_function(x_k_minus) - OCV_function(x_k_minus_1, V_measured) - (1 - x_k_minus + x_k_minus_1) * R0) / (x_k_minus - x_k_minus_1)
    #    R1 = R1_new
    #    x = None
    #    if (1 - x_k_minus_1) == 0:
    #        #to prevent it being divided by zero and erroring
    #        x = 1
    #    else:
    #        x = (1 - x_k_minus_1)
    #    R0_new = (OCV_function(x_k_minus_1) - R1 + (1 - x_k_minus_1) * R0) / x
    #    x = OCV_function(x_k_minus_1) - R1
    
    #V2 = R0_new *  0.14 #Vin divided by Imin
    #V2 = float(V2)
    #V1 = V_measured - V2 
    #R1 = V1 / 0.16 #Vin divided by Imax

    R1 = 6.955
    R0 = 6.955
    return R1, R0

def g_function(x_k_minus, x_k_minus_1, V_measured):
    #used to calculate the predicted terminal voltage (at the inital estimate for battery SOC). This is then compared to the actual measured terminal voltage, and will help determine the Kalman gain, which is responsible for weighing the 2 possible answers (one based on measured data, one based on ML) to return the most accurate answer.  
    #take voltage measurement from user (in my case it is being extracted from the google sheets data), and use to compare and calculate values. The idea is that I am relating OCV (open circuit voltage, from my SOC-OCV curve) to terminal voltage. These will be different. 
    R1, R0 = calculate_R1_R0()
    
    R_total = R1 + R0
    #calculating current across full circuit; defining i
    i = float(V_measured) / R_total

    #assuming that voltage across R1 is the same as the voltage across the circuit, since they are connected in series.
    i_R1 = float(V_measured)/ R1

    #returns expected terminal voltage
    return OCV_function(x_k_minus) - R1 * i_R1 - R0 * i

#MATRICIES
#note - these are technically supposed to be matricies, but I've defined my result (SOC) as a scalar value. Therefore I've represented everything, including variables named _matrix, as an integer. This also decreases complexity. 
def calculate_A_matrix(x_k_minus, x_k_minus_1):
    #equation = f_function('x_k_minus', 'x_k_minus_1')
    #the equation to calculate A matrix is partial derivative of f function with respect to x_k_minus_1. So it is equivalent to x_k_minus, or predicted value at time k, over x_k_minus_1, estimate at previous timestep. 
    #because I am taking the derivative, taking the derivative of two values returns 0, which then serves as a zero multiplier throughout the rest of the code. 
    #in order to avoid this, I am calculating the slope (rise/run) instead of the derivative. Conceptually, they are the same, especially because I have such a large number of datapoints, therefore the error is negligeable. 
    result = x_k_minus - x_k_minus_1 / 1

    A_matrix = result * -1 #so it isn't negative

    return A_matrix

#B matrix is technically in the sheet of equations, but it is not actually called with the way I've structured my code - I've left out a lot of error terms for simplicity. 
#def calculate_B_matrix(x_k_minus, x_k_minus_1):
#    f = f_function(x_k_minus, x_k_minus_1) + wk
#    B_matrix = sp.diff(f, wk)
#    return B_matrix


def calculate_C_matrix(x_k_minus, x_k_minus_1, V_measured, vk):
    # Define symbolic variable for x_k_minus. the derivative function only works with symbolic variables. 
    x_k_minus_symbolic = sp.symbols('x_k_minus_symbolic')
    # Calculate g
    g = g_function(x_k_minus_symbolic, x_k_minus_1, V_measured) + vk
    # Calculate the derivative of g with respect to x_k_minus
    C_matrix = sp.diff(g, x_k_minus_symbolic)
    # Evaluate the derivative at the given value of x_k_minus
    C_matrix_value = C_matrix.evalf(subs={x_k_minus_symbolic: x_k_minus})
    return C_matrix_value

#the D matrix is also not called in the rest of my code
#def calculate_D_matrix(x_k_minus):
#    g = g_function(x_k_minus, k) + vk
#   D_matrix = sp.diff(g, vk)
#    return D_matrix

    
def plot_SOC(estimations, intervals): 
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

    #error matrices and terms
    sigma_k_minus_1 = 1.00 #initalizing error covariance matrix as equivalent to identify matrix
    x_k_minus_1 =  100 #final guess for previous timestep (my data starts as fully charged, therefore 100% SOC)
    vk_scalar = 0.0 #this is an error term but I took it out for simplicity. 
    wk = 0 #noise 
    vk = 1 #noise
    scale_factor = 1e12 #used to avoid nan values. A lot of these values are very small numbers, so python will round them to zero or have trouble performing operations with them, and that will cause errors in my code. 
    
    #measured values (extracted from google sheet later)
    i_measured = 0.0
    V_measured = 0.0

    #adding returned values to lists so data can be plotted afterwards
    estimations = []
    intervals = []

    #for sake of simplicity, I am setting vk = 1 and ignoring the effects of noise. 
    """
    # calculating variance of Hall Effect sensor noise. standard deviation = error x callibration factor. it is a gaussian distribution with a zero mean
    mean = 0
    std_dev = 0.05 * 2.76854928e-4
    vk = Normal("vk", mean, std_dev)

    # Sample a value from the normal distribution to get vk_scalar
    #vk_scalar = np.random.normal(mean, std_dev)
    vk_scalar = vk_scalar * std_dev + mean  # Adjust vk_scalar to match the mean and standard deviation of the distribution
    """
    #wk represents the process noise error. it has a zero mean, and is given by E(wk*wk)^T = R_k, where R_k is the covariance matrix with the process noise. this is basically any unaccounted error or discrepencies in the system dynamics. Everything is defined by only one dimension, therefore calculating a covariance matrix doesn't make sense. So I will set wk to zero, because of the zero mean 
    
    for k in range(1, 100): #put to 17274 after it works
        #this iterates through the actual extended kalman filter equations, and returns the error term as well as the predicted SOC value for that timestep (ie, iteration number)

        #extract current and voltage value from google sheet to use in calculations
        row_index = k + 1
        i_measured = float(sheet.cell(row_index, 3).value)
        V_measured = float(sheet.cell(row_index, 2).value)

        #calculate x_k_minus_1 (initial guess) with state space model
        x_k_minus = f_function(x_k_minus_1, i_measured)
    
        #calculating A and C matrices breforehand to decrease computational time (ie they are calculated 1 time per iteration, not 5 times)
        C_matrix = calculate_C_matrix(x_k_minus, x_k_minus_1, V_measured, vk)
        A_matrix = calculate_A_matrix(x_k_minus, x_k_minus_1)

        #scaling up to account for the numbers being super small. so python can handle them. 
        A_matrix_large = A_matrix * scale_factor
        sigma_k_minus_1_large = sigma_k_minus_1 * scale_factor

        #calculate error covariance time update
        sigma_k_minus_large =  float(A_matrix_large * sigma_k_minus_1_large * A_matrix_large)
        sigma_k_minus = sigma_k_minus_large / scale_factor
        print("sigma_k_minus:", sigma_k_minus)

        #calculate kalman gain - weights my two answers according to past error term (ie, how accurate it's past estimates have been)
        Kalman_gain = sigma_k_minus * C_matrix * ((C_matrix * sigma_k_minus * C_matrix + vk_scalar)**-1)
        
        #state estimate measuremnet update based on kalman gain
        R1, R0 = calculate_R1_R0()
        i_R1 = V_measured/ R1


        #OCV_result = OCV_function(x_k_minus)
        #result = OCV_result - R1 * i_R1 - R0 * i_measured
        #result is on the order of 50, likely because R1/R0 values are incorrect. 
        result = OCV_function(x_k_minus) - R1 * i_R1 - R0 * i_measured

        #yk is the expected terminal voltage
        yk = g_function(x_k_minus, x_k_minus_1, V_measured)
        
        #calculating final estimation for SOC at time k 
        x_k_plus = x_k_minus + Kalman_gain * (yk - result) 
        
        #scaling up values to ensure precision when multiplying very small numbers
        Kalman_gain_large = Kalman_gain * scale_factor
        C_matrix_large = C_matrix * scale_factor
        sigma_k_minus_large = sigma_k_minus * scale_factor 

        #calculating the error covariance matrix, used to computer values in next timestep. 
        sigma_k_plus_large = (scale_factor - Kalman_gain_large * C_matrix_large) * sigma_k_minus_large 
        sigma_k_plus_calc = sigma_k_plus_large / scale_factor

        #scaling down sigma_k_plus to avoid nan results
        if k / 10 == 0: 
            x = k * 10
            sigma_k_plus_new = sigma_k_plus / (10 ** x)
        else: 
            sigma_k_plus_new = sigma_k_plus_calc / 1e40
        #print("sigma_k_plus:", sigma_k_plus)

        print("interval: {}".format(k))
        print("x_k_plus: {}, sigma_k_plus: {}".format(x_k_plus, sigma_k_plus_new))

        k += 1

        estimations.append(x_k_plus)
        intervals.append(k)

        #updating variables to prepare for next timestep
        x_k_minus_1 = x_k_plus
        sigma_k_minus_1 = sigma_k_plus_new
        sigma_k_plus = 1

        #adding delay to account for the API limit of 300 requests/minute 
        time.sleep(0.6)

    plot_SOC(estimations, intervals)

    return estimations, intervals
    
main()

import math
from scipy import stats
import statistics
'''instead of basing test selection on inputted variables
I'm gonna use a flowchart
it should be easier and more direct
because the prior system could get
very messy once proportions and 2 samples are introduced'''

'''ran into error with scipy functions below "scipy.stats.norm.ppf"
had to change import at top from import scipy to from scipy import stats
and also delete scipy.stats.etc and deleted scipy and instead put stats.etc, '''

#questionaire variables to discriminate what inference to use
#inference using proportions, means, or slope (s)?
proportion = False
mean = False
linear_regression = False
chi_squared = False

z = False
chi_type = False

#defining the sample function variables outside the function to have them globally defined for the inferencing choosing if block
one_sample = False
two_sample = False
paired_sample = False
#defining the test/interval function variables outside the function to have them globally defined for the inferencing choosing if block
interval = False
test = False


#sample questionaire function inference using 1 sample, 2 samples, or paired/matched?
def sample_questionaire():
    global one_sample
    one_sample = False
    global two_sample
    two_sample = False
    global paired_sample
    paired_sample = False

    sample_number = input("How many samples? 1, 2, or paired (p): ")
    if sample_number == "1":
        one_sample = True
    elif sample_number == "2":
        two_sample = True
    elif sample_number == "p":
        paired_sample = True


#interval or test questionaire function?
def interval_test_questionaire():
    global interval
    interval = False
    global test
    test = False
    interval_or_test = input("Is it an interval (i) looking for a range or a test for a hypothesis(t)?: ")
    if interval_or_test == "i":
        interval = True
    elif interval_or_test == "t":
        test = True


'''
#using global module
from Globals import alpha_tail_function
alpha_tail_function()
'''


alpha_level = float(input("What is the significance/alpha level?: "))
confidence_level = 1 - alpha_level
tailed = input("Is it 1 or 2 tailed i.e. looking for greater (g), lesser (l) or difference (d)?: ")
#if Ha > x, then right tailed
#if Ha < x, then left tailed
#if Ha =! x, then two tailed, alpha/2
#save value as a variable, to then use universally for any inference later on

# critical value script

alpha_tail = 0
alternative = ""
if tailed == "l":
    alpha_tail = alpha_level
    alternative = "less"
elif tailed == "g":
    alpha_tail = 1 - alpha_level
    alternative = "two-sided"
elif tailed == "d" or "1":
    alpha_tail = 1 - alpha_level / 2
    alternative = "great"


#critical value functions
#one samples
def z_cv_function():
    global z_critical_value
    z_critical_value = stats.norm.ppf(alpha_tail)
    print(f"Z Critical Value: {z_critical_value}")


def one_sample_t_cv_function():
    global sample_size
    sample_size = float(input("Sample Size?: "))
    degrees_of_freedom = sample_size - 1
    global t_critical_value
    t_critical_value = stats.t.ppf(q=alpha_tail, df=degrees_of_freedom)
    print(f"T Critical Value: {t_critical_value}")


#two samples
def two_sample_t_cv_function():
    global n1
    global n2
    n1 = float(input("Sample 1 Size?: "))
    n2 = float(input("Sample 2 Size?: "))
    degrees_of_freedom = n1 - 1 + n2 - 1
    global t_critical_value
    t_critical_value = stats.t.ppf(q=alpha_tail, df=degrees_of_freedom)
    print(f"T Critical Value: {t_critical_value}")





#Inference Questionaire
inference_parameter_questionaire = input("Does the inference use proportions (p), means (x), slope (s), or categories (c)?: ")
if inference_parameter_questionaire == "p":
    proportion = True
    sample_questionaire()
    interval_test_questionaire()
    #must be z
elif inference_parameter_questionaire == "x":
    mean = True
    #can be either z or t, depends if pop SD (o) is known
    o_known = input("Is the Population SD (o) known? Yes (y) or No (n): ")
    if o_known == "y":
        z = True
    sample_questionaire()
    interval_test_questionaire()
elif inference_parameter_questionaire == "s":
    linear_regression = True
elif inference_parameter_questionaire == "c":
    chi_squared = True
    #chi_type = input("Are you looking for an association (a), a comparison to expected frequencies (b), or same distribution (c)?: ")




#individual inference modules

def one_sample_z_test_prop():
    population_proportion = float(input("Population Proportion?: "))
    successes = float(input("# of successes: "))
    sample_size = float(input("# of trials/sample size: "))
    sample_proportion = successes / sample_size
    z = (sample_proportion - population_proportion)/math.sqrt((population_proportion * (1 - population_proportion))/sample_size)
    if abs(z) >= z_critical_value:
        print(f"""The Z is {z}, which is outside the Z score bounds of +-{z_critical_value}
    therefore, we reject the null hypothesis""")
    elif abs(z) < z_critical_value:
        print(f"""The Z is {z} which lies inside the Z score bounds of +-{z_critical_value};
    therefore, we accept the null hypothesis""")


def one_sample_z_interval_prop():
    successes = float(input("# of successes: "))
    n = float(input("# of trials/sample size: "))
    p = successes / n
    condition = n*p * (1 - p)
    if condition >= 10:
        upperbound = p + z_critical_value * math.sqrt((p * (1 - p))/n)
        lowerbound = p - z_critical_value * math.sqrt((p * (1 - p))/n)
        print(f"""The sample proportion is {p} and we are {confidence_level}% confident
the true population proportion lies within ({lowerbound}, {upperbound})""")
    elif condition < 10:
        print("np(1 - p) >= 10 condition not met for assuming normal distribution :(")


def one_sample_z_test_mean():
    population_mean = float(input("Population Mean?: "))
    population_sd = float(input("Population SD?: "))
    sample_mean = float(input("Sample Mean?: "))
    sample_size = float(input("Sample Size?: "))
    print("Performing 1 Sample Z Test...")
    Z = (sample_mean - population_mean) / (population_sd / math.sqrt(sample_size))
    if abs(Z) >= z_critical_value:
        print(f"""The Z is {Z}, which is outside the Z score bounds of +-{z_critical_value}
therefore, we reject the null hypothesis""")
    elif abs(Z) < z_critical_value:
        print(f"""The Z is {Z} which lies inside the Z score bounds of +-{z_critical_value};
therefore, we accept the null hypothesis""")


def one_sample_z_interval_mean():
    o = float(input("Population SD?: "))
    x = float(input("Sample Mean?: "))
    n = float(input("Sample Size?: "))
    confidence_interval_upperbound = x + z_critical_value * o / math.sqrt(n)
    confidence_interval_lowerbound = x - z_critical_value * o / math.sqrt(n)
    print(f"We are {confidence_level}% confident that the population mean score")
    print(f"is between ({confidence_interval_lowerbound}, {confidence_interval_upperbound})")


def one_sample_t_test_mean():
    population_mean = float(input("Population Mean?: "))
    sample_mean = float(input("Sample Mean?: "))
    sample_sd = float(input("Sample SD?: "))
    # t value
    t_value = (sample_mean - population_mean) / (sample_sd / math.sqrt(sample_size))
    print(f"T Value = {t_value}")
    if abs(t_value) >= t_critical_value:
        print(f"""The data's t-value of {t_value}
    is outside the bounds of the critical value of {t_critical_value} and {-t_critical_value}.
    Therefore, there IS EVIDENCE the experiment had a statistically significant affect.""")
    elif abs(t_value) < t_critical_value:
        print(f"""The data's t-value of {t_value}
    is inside the bounds of the critical value of {t_critical_value} and {-t_critical_value}.
    Therefore, there is NO EVIDENCE the experiment had a statistically significant affect.""")


def one_sample_t_interval_mean():
    x = float(input("Sample Mean?: "))
    sample_sd = float(input("Sample SD?: "))
    confidence_interval_upperbound = x + t_critical_value * sample_sd / math.sqrt(sample_size)
    confidence_interval_lowerbound = x - t_critical_value * sample_sd / math.sqrt(sample_size)
    print(f"""We are {confidence_level}% the population mean lies within
({confidence_interval_lowerbound}, {confidence_interval_upperbound})""")


def two_sample_z_test_proportion():
    x1 = float(input("Sample 1 Successes?: "))
    n1 = float(input("Sample 1 Size?: "))
    x2 = float(input("Sample 2 Successes?: "))
    n2 = float(input("Sample 2 Successes?: "))
    p1 = x1 / n1
    p2 = x2 / n2
    p = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / ((math.sqrt(p * (1 - p))) * (math.sqrt((1 / n1 + 1 / n2))))
    if abs(z) >= z_critical_value:
        print(f"""The Z is {z}, which is outside the Z score bounds of +-{z_critical_value}
    therefore, there is a statistically significant difference between the two groups""")
    elif abs(z) < z_critical_value:
        print(f"""The Z is {z} which lies inside the Z score bounds of +-{z_critical_value};
    therefore, there is no statistically significant difference between the two groups""")


def two_sample_z_interval_proportions():
    x1 = float(input("Sample 1 Successes?: "))
    n1 = float(input("Sample 1 Size?: "))
    x2 = float(input("Sample 2 Successes?: "))
    n2 = float(input("Sample 2 Successes?: "))
    p1 = x1 / n1
    p2 = x2 / n2
    p = (x1 + x2) / (n1 + n2)
    upperbound = (p1 - p2) + z_critical_value * math.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
    lowerbound = (p1 - p2) - z_critical_value * math.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
    print(f"""The sample proportion is {p} and we are {confidence_level}% confident
the true population proportion lies within ({abs(lowerbound)}, {abs(upperbound)})""")


def two_sample_z_test_mean():
    x1 = float(input("Sample 1 Mean?: "))
    s1 = float(input("Sample 1 SD?: "))
    n1 = float(input("Sample 1 Size?: "))
    x2 = float(input("Sample 2 Mean?: "))
    s2 = float(input("Sample 2 SD?: "))
    n2 = float(input("Sample 2 Size?: "))
    z = (x1 - x2) / (math.sqrt(s1 * s1 / n1 + s2 * s2 / n2))
    if abs(z) >= z_critical_value:
        print(f"""The Z is {z}, which is outside the Z score bounds of +-{z_critical_value}
    therefore, there is a statistically significant difference between the two groups""")
    elif abs(z) < z_critical_value:
        print(f"""The Z is {z} which lies inside the Z score bounds of +-{z_critical_value};
    therefore, there is no statistically significant difference between the two groups""")


def two_sample_t_test_mean():
    x1 = float(input("Sample 1 Mean?: "))
    s1 = float(input("Sample 1 SD?: "))
    x2 = float(input("Sample 2 Mean?: "))
    s2 = float(input("Sample 2 SD?: "))
    df1 = n1 - 1
    df2 = n2 - 1
    ss1 = s1 * s1 * df1
    ss2 = s2 * s2 * df2
    sp = (ss1 + ss2) / (df1 + df2)
    t_value = (x1 - x2) / (math.sqrt(sp * sp / n1 + sp * sp / n2))
    if abs(t_value) >= t_critical_value:
        print(f"""The data's t-value of {t_value}
    is outside the bounds of the critical value of +-{t_critical_value}.
    Therefore, there IS EVIDENCE the means of the samples are statistically different.""")
    elif abs(t_value) < t_critical_value:
        print(f"""The data's t-value of {t_value}
    is inside the bounds of the critical value of {t_critical_value} and {-t_critical_value}.
    Therefore, there is NO EVIDENCE the means of the samples are statistically different.""")


def two_sample_t_interval_mean():
    x1 = float(input("Sample 1 Mean?: "))
    s1 = float(input("Sample 1 SD?: "))
    x2 = float(input("Sample 2 Mean?: "))
    s2 = float(input("Sample 2 SD?: "))
    lowerbound = x1 - x2 - (t_critical_value * (math.sqrt(s1 * s1 / n1 + s2 * s2 / n2)))
    upperbound = x1 - x2 + (t_critical_value * (math.sqrt(s1 * s1 / n1 + s2 * s2 / n2)))
    print(f"""We are {confidence_level}% confident that the true difference between means
is between ({lowerbound}, {upperbound})""")


#wrong paired t test???
def false_paired_t_test_mean():
    pre_data = input("Type all pre-treatment data with a space in between: ")
    pre_x = [float(x) for x in pre_data.split()]
    post_data = input("Type all post-treatment data with a space in between (in same order as pre-treatment): ")
    post_x = [float(x) for x in post_data.split()]
    pre_total = 0.00
    for pre_i in pre_x:
        pre_total += pre_i
    print(f"pre total {pre_total}")
    post_total = 0.00
    for post_i in post_x:
        post_total += post_i
    print(f"post total {post_total}")
    pre_mean = pre_total / len(pre_data)
    post_mean = post_total / len(post_data)
    diff_mean = post_mean - pre_mean
    x_diff_list = list()
    for pre_i, post_i in zip(pre_x, post_x):
        x_diff_list.append(pre_i - post_i)
    print("X Difference: ", x_diff_list)
    x_squared_total = 0.00
    for x in x_diff_list:
        x_squared_total += x*x
    print(x_squared_total)
    sd = math.sqrt((x_squared_total - (diff_mean * diff_mean / sample_size)) / (sample_size - 1))
    print(f"SD {sd}")
    t = diff_mean / (sd / math.sqrt(len(pre_data)))
    if abs(t) > t_critical_value:
        print(f"""The data's t value of {t} is outside the bounds of
+-{t_critical_value}; therefore, there is evidence for
statistically significant difference from treatment""")
    if abs(t) < t_critical_value:
        print(f"""The data's t value of {t} is within the bounds of
        +-{t_critical_value}; therefore, there is NO evidence
of a statistically significant difference from treatment""")


def paired_t_test_mean():
    pre_data = input("Type all pre-treatment data with a space in between: ")
    pre_x = [float(x) for x in pre_data.split()]
    post_data = input("Type all post-treatment data with a space in between (in same order as pre-treatment): ")
    post_x = [float(x) for x in post_data.split()]
    x_diff_list = list()
    for pre_i, post_i in zip(pre_x, post_x):
        x_diff_list.append(pre_i - post_i)
    mean = statistics.mean(x_diff_list)
    sd = stats.tstd(x_diff_list)
    t = mean / (sd / math.sqrt(sample_size))
    if abs(t) >= t_critical_value:
        print(f"""The data's t-value of {t}
    is outside the bounds of the critical value of {t_critical_value} and {-t_critical_value}.
    Therefore, there IS EVIDENCE the experiment had a statistically significant affect.""")
    elif abs(t) < t_critical_value:
        print(f"""The data's t-value of {t}
    is inside the bounds of the critical value of {t_critical_value} and {-t_critical_value}.
    Therefore, there is NO EVIDENCE the experiment had a statistically significant affect.""")


def paired_t_interval_mean():
    pre_data = input("Type all pre-treatment data with a space in between: ")
    pre_x = [float(x) for x in pre_data.split()]
    pre_x_mean = statistics.mean(pre_x)
    post_data = input("Type all post-treatment data with a space in between (in same order as pre-treatment): ")
    post_x = [float(x) for x in post_data.split()]
    post_x_mean = statistics.mean(post_x)
    x_diff_list = list()
    for pre_i, post_i in zip(pre_x, post_x):
        x_diff_list.append(pre_i - post_i)
    mean = statistics.mean(x_diff_list)
    sd = stats.tstd(x_diff_list)
    lowerbound = (post_x_mean - pre_x_mean) - t_critical_value * sd / math.sqrt(sample_size)
    upperbound = (post_x_mean - pre_x_mean) + t_critical_value * sd / math.sqrt(sample_size)
    print(f"""We are {confidence_level}% confident that the true difference between means
    is between ({lowerbound}, {upperbound})""")

def linear_regression_test():
    x_coords = input("Type all x coordinates with a space in between: ")
    x_data = [float(x) for x in x_coords.split()]
    y_coords = input("Type all y coordinates in same order with a space in between: ")
    y_data = [float(y) for y in y_coords.split()]
    linear = stats.linregress(x_data, y_data)
    print(f"""Linear Regression: y = {linear.slope}x +{linear.intercept}
{linear}""")


def chi_square():
    obs = input("Type in observed values with a space in between: ")
    obs = [float(x) for x in obs.split()]
    exp = input("Type in expected values in same order with a space in between: ")
    exp = [float(x) for x in exp.split()]
    dof = len(obs) - 1
    chi = stats.chisquare(obs, exp, dof)
    print(chi)
    t_critical_value = stats.t.ppf(q=alpha_tail, df=dof)
    if chi.statistic > t_critical_value:
        print(f"""The data is statistically significant different than the expected
since the x^2 = {chi.statistic}, which is greater than the T critical value of {t_critical_value}""")





#inference choosing if block
if one_sample and test and proportion:
    print("1 sample z test for proportions")
    z_cv_function()
    one_sample_z_test_prop()
elif one_sample and interval and proportion:
    print("1 sample z interval for proportions/Confidence Interval for Population Proportions")
    z_cv_function()
    one_sample_z_interval_prop()
elif one_sample and z and test and mean:
    print("1 sample z test for mean")
    z_cv_function()
    one_sample_z_test_mean()
elif one_sample and z and interval and mean:
    print("1 sample z interval for mean")
    z_cv_function()
    one_sample_z_interval_mean()
elif one_sample and z == False and test and mean:
    print("1 sample t test for mean")
    one_sample_t_cv_function()
    one_sample_t_test_mean()
elif one_sample and z == False and interval and mean:
    print("1 sample t interval for mean")
    one_sample_t_cv_function()
    one_sample_t_interval_mean()
elif two_sample and test and proportion:
    print("2 sample z test for proportions")
    z_cv_function()
    two_sample_z_test_proportion()
elif two_sample and interval and proportion:
    print("2 sample z interval for proportions")
    z_cv_function()
    two_sample_z_interval_proportions()
elif two_sample and z and test and mean:
    print("2 sample z test for means")
    z_cv_function()
    two_sample_z_test_mean()
elif two_sample and z and interval and mean:
    print("2 sample z interval for means")
    '''does 2 sample z t/i for means not exist???'''
elif two_sample and z == False and test and mean:
    print("Independent/2 sample t test for means")
    two_sample_t_cv_function()
    two_sample_t_test_mean()
elif two_sample and z == False and interval and mean:
    print("Independent/2 sample t interval for means")
    two_sample_t_cv_function()
    two_sample_t_interval_mean()
elif paired_sample and test and proportion:
    print("Dependent/Paired sample z test for proportions")
    '''does paired z test/interval also not exist???'''
elif paired_sample and interval and proportion:
    print("Dependent/Paired sample z test for proportions")
elif paired_sample and z and test and mean:
    print("Dependent/Paired sample z test for mean")
elif paired_sample and z and interval and mean:
    print("Dependent/Paired z interval for means")
elif paired_sample and z == False and test and mean:
    print("Dependent/Paired sample t test for means")
    one_sample_t_cv_function()
    paired_t_test_mean()
elif paired_sample and z == False and interval and mean:
    print("Dependent/Paired sample t interval for means")
    one_sample_t_cv_function()
    paired_t_interval_mean()
elif linear_regression:
    print("Linear Regression")
    linear_regression_test()
elif chi_squared:
    print("Chi Square Test of Independence/Association")
    chi_square()
elif chi_type == "b":
    print("Chi Square Goodness of Fit")
elif chi_type == "c":
    print("Chi Square Test of Homogeneity")

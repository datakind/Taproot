import time, os, operator
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.neighbors.kde import KernelDensity
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, r2_score

PROJECT_ROOT = os.path.dirname(__file__)

def replaceNA(training_df, cv_df, excludedCols):
	naLabels = []
	naReplacementValues = []
	for feat in training_df.columns:
		if feat not in excludedCols:
			# means of null and non-null values
			naRepayment = np.mean(training_df.ix[pd.isnull(training_df[feat]), 'outcome'])
			notNaRepayment = np.mean(training_df.ix[pd.notnull(training_df[feat]), 'outcome'])
			goodValue = np.mean(training_df.ix[(pd.notnull(training_df[feat])) & (training_df['outcome'] == 1), feat])
			badValue = np.mean(training_df.ix[(pd.notnull(training_df[feat])) & (training_df['outcome'] == 0), feat])
			if naRepayment > notNaRepayment:
				# if NA mean repayment bigger, replace NA values with mean of good training values
				naReplacement = goodValue
			else:
				naReplacement = badValue
			training_df.ix[pd.isnull(training_df[feat]), feat] = naReplacement
			cv_df.ix[pd.isnull(cv_df[feat]), feat] = naReplacement
			naLabels.append(feat)
			naReplacementValues.append(naReplacement)
	return naLabels, naReplacementValues

def convertAnswer2Int(val):
	if val == 'Strongly Disagree':
		return 1
	elif val == 'Strongly Agree':
		return 2
	elif val == 'Disagree Somewhat':
		return 3
	elif val == 'Disagree':
		return 4
	elif val == 'Agree Somewhat':
		return 5
	elif val == 'Agree':
		return 6
	else:
		if pd.notnull(val):
			print "bad value: %s" % val
		return None

def transformFeature(feature, transform):
	if transform == 'log':
		if feature is None:
			transformedFeature = None
		else:
			transformedFeature = np.log(feature + 1)
	return transformedFeature

def plotCategoricalFeatures(featureCols, evalMetrics):
	for categoricalCol in featureCols:
		if categoricalCol == 'assigned_grant_type_normalized':
			tmp_df = projects_df.ix[projects_df[categoricalCol].isin(['HR','Annual Report','Financial Analysis','Naming, Visual Identify, and Brand']), :]
			df = pd.DataFrame(index=tmp_df.groupby([categoricalCol]).mean().index)
		else:
			df = pd.DataFrame(index=projects_df.groupby([categoricalCol]).mean().index)
		for col in evalMetrics:
			if categoricalCol == 'assigned_grant_type_normalized':
				tmp_df = projects_df.ix[projects_df[categoricalCol].isin(['HR','Annual Report','Financial Analysis','Naming, Visual Identify, and Brand']), :]
				df[col] = tmp_df.groupby([categoricalCol])[col].mean().tolist()
			else:
				df[col] = projects_df.ix[pd.notnull(projects_df[categoricalCol]),:].groupby([categoricalCol])[col].mean().tolist()

		# plotting satisfaction vs assigned_grant_type
		plt.figure()
		ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
		df.plot(ax=ax, kind='bar', fontsize=20)
		ax.set_ylim([4, 6])
		patches, labels = ax.get_legend_handles_labels()
		ax.legend(patches, labels, loc='best', fontsize=14)
		plt.xlabel(ax.get_xlabel(), fontsize=20)
		plt.show()

def plotContinuousFeatures(featureCols, evalMetrics):
	for col in featureCols:
		print col
		for evalMetric in evalMetrics:
			print evalMetric
			good = projects_df[projects_df[evalMetric] == 6][col].dropna().reshape(-1, 1)
			bad = projects_df[projects_df[evalMetric] < 6][col].dropna().reshape(-1, 1)

			# calculate percent repayment for NA, not-NA
			naMean = np.mean(projects_df.ix[pd.isnull(projects_df[col]), evalMetric].dropna())
			not_naMean = np.mean(projects_df.ix[pd.notnull(projects_df[col]), evalMetric].dropna())
			na_df = pd.DataFrame([naMean, not_naMean], index=['NA', 'not NA'])

			# plot
			colMin = np.nanmin(np.concatenate([good, bad]))
			colMax = np.nanmax(np.concatenate([good, bad]))
			bwidth = (colMax - colMin) / float(10)
			x = np.linspace(colMin, colMax, 100).reshape(-1, 1)

			good_kde = KernelDensity(bandwidth=bwidth).fit(good)
			good_density = np.exp(good_kde.score_samples(x))

			bad_kde = KernelDensity(bandwidth=bwidth).fit(bad)
			bad_density = np.exp(bad_kde.score_samples(x))

			maxDensity = np.nanmax(np.concatenate([good_density, bad_density]))

			plt.figure()
			ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
			plt.plot(x, good_density, 'g')
			plt.fill_between(np.linspace(colMin, colMax, 100), good_density, facecolor='g', alpha=.2)
			plt.plot(x, bad_density, 'r')
			plt.fill_between(np.linspace(colMin, colMax, 100), bad_density, facecolor='r', alpha=.2)
			plt.plot(good, good * 0 + 1.050*maxDensity, 'og', alpha=.1)
			plt.plot(bad, bad * 0 + 1.075*maxDensity, 'or', alpha=.1)
			plt.xlabel(col)
			plt.ylabel("Density")
			plt.ylim([0, 1.1*maxDensity])
			plt.plot((np.median(good), np.median(good)), (1.0*maxDensity, 1.1*maxDensity), 'g', label='good median')
			plt.plot((np.median(bad), np.median(bad)), (1.0*maxDensity, 1.1*maxDensity), 'r', label='bad median')
			# plt.legend(loc="lower center")
			plt.legend(loc="best")
			text = "%.0f Very Satisfied\n%.0f Not Very Satisfied" % (len(good), len(bad))
			# ax.text(.15*x[-1],1*maxDensity, text, horizontalalignment="left", verticalalignment="top")
			plt.title("%s\n%s" % (text, evalMetric))

			ax = plt.subplot2grid((1, 4), (0, 3), colspan=1)
			plt.bar([.25, .75], [naMean, not_naMean], width=.4, align='center')
			plt.xticks([.25, .75], ['NA', 'not NA'])
			plt.ylim([4, 6])
			plt.ylabel('Mean Satisfaction')
			ax.yaxis.tick_right()
			ax.yaxis.set_label_position("right")
			text = "%.0f NA values\n%.0f non-NA values" % ((len(projects_df[evalMetric]) - len(good) - len(bad)) ,(len(good) + len(bad)))
			ax.text(.1,1, text, horizontalalignment="left", verticalalignment="top")
			plt.show()

if __name__ == '__main__':
	startTime = time.time()

	plotFlag = True
	localPath = os.path.join('..', 'Data')

	# read in data
	projApp2projEval_link_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Application-Project Evaluation links.csv'))
	projectsApps_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Applications.csv'))
	projectsAwarded_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Awarded.csv'))
	projectsEvals_df = pd.read_csv(os.path.join(PROJECT_ROOT, '..', 'Codes', 'Output', 'NPO_Eval_2010Q2.csv'))
	proj2vol_link_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects-Volunteers Links.csv'))
	volunteersApp_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Volunteers Applications.csv'))
	volunteersEvals_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Volunteers Evaluation (2010 - Q2).csv'))

	# there are 60 duplicate emails in projectsEvals
	duplicateEmails = []
	for email, c in zip(projectsEvals_df.groupby(['Email Address'])['Email Address'].count().index,
		projectsEvals_df.groupby(['Email Address'])['Email Address'].count()):
		if c > 1:
			duplicateEmails.append([email, c])

	# do any emails show up with multiple org_names? (NO: every email is associated with a single org name!)
	emails = defaultdict(list)
	for row in np.array(projApp2projEval_link_df):
		org_name = row[1]
		email = row[-1]
		if email not in emails:
			emails[email].append(org_name)
	for email, org_names in emails.items():
		if len(set(org_names)) > 1:
			print email, org_names

	# remove duplicate emails from projApp2projEval_link_df
	# unique key = 'app contact email'
	tmp_array = []
	emails = {}
	for row in np.array(projApp2projEval_link_df):
		email = row[-1]
		if email not in emails:
			emails[email] = None
			tmp_array.append(row)
	projApp2projEval_link_df = pd.DataFrame(np.array(tmp_array), columns=projApp2projEval_link_df.columns)

	# combine dataframes
	projectsComb_df = pd.merge(projectsApps_df, projectsAwarded_df, left_on=['sgapp_id', 'org_id'], 
		right_on=['sgapp_id', 'org_id'], how='left')
	# 1200 evals
	projectsAppEvalComb_df = pd.merge(projApp2projEval_link_df, projectsEvals_df,
		left_on='app contact email', right_on='Email Address', how='inner')
	# NOTE: some emails in the evaluation file don't match with application-evaluation-links file

	# use this file for feature building
	projects_df = pd.merge(projectsComb_df, projectsAppEvalComb_df, left_on='sgapp_id', right_on='sgaid', how='inner')

	# CLEAN DATA
	# remove bad employees_ft
	projects_df.ix[projects_df['employees_ft'] >= 1000000, 'employees_ft'] = None

	# add outcome label
	projects_df['outcome'] = None
	projects_df.ix[projects_df['How satisfied with the overall experience? '] < 6, 'outcome'] = 0
	projects_df.ix[projects_df['How satisfied with the overall experience? '] == 6, 'outcome'] = 1

	# for each ord_name, choose the first entry. This way, we don't bias the model.
	
	# find the earliest timestamp for each org_id
	org_ids = {}
	for row in np.array(projects_df):
		org_id = row[1]
		date_app_completed = row[4]
		if org_id not in org_ids:
			org_ids[org_id] = pd.to_datetime(date_app_completed)
		elif org_ids[org_id] > pd.to_datetime(date_app_completed):
			org_ids[org_id] = pd.to_datetime(date_app_completed)
	
	# if timestamp matches, and org_id not in array
	tmp_ids = {}
	tmp_array = []
	for row in np.array(projects_df):
		org_id = row[1]
		date_app_completed = row[4]
		if org_id not in tmp_ids and pd.to_datetime(date_app_completed) == org_ids[org_id]:
			tmp_ids[org_id] = None
			tmp_array.append(row)
	projects_df = pd.DataFrame(np.array(tmp_array), columns=projects_df.columns)

	projects_df.to_csv(os.path.join(PROJECT_ROOT, localPath, 'projects.csv'), index=False)

	##################

	# clustering idea
	# 1) size
	# 2) purpose of grant
	# 3) support of organization (have a plan, have high-level support)
	# 4) quality of application

	satisfactionCols = ['How satisfied with the overall experience? ', "How satisfied with the final deliverable? ", "How satisfied with the Taproot Team (e.g. great to work with, professional, capable)", "How satisfied with the support provided by the Taproot Foundation during the project (e.g. attention from staff, blueprint materials) ", "How satisfied with how smoothly and on-time the project ran? ", "How satisfied with how long the overall project took? ", "How satisfied with the amount of time the project required of you? "]
	believeCols = ["I believe that the Service Grant deliverables has/will strengthen our organization's infrastructure or resources","I believe that the Service Grant process has provided or will provide significant learning to me or our management or staff","I believe that the Service Grant process has or will positively change management or staff's attitudes or beliefs (or the organization's culture)","I believe that the Service Grant has led or will lead to positive changes in the activities undertaken by the organization or key people","I believe that taken together the Service Grant will have a significantly positive impact on our organization's management or governance capability (includes management, fundraising, communications, leadership, human resource development, community outreach, etc.)"]

	for col in believeCols:
		projects_df[col] = projects_df[col].apply(convertAnswer2Int)
	metricCols = satisfactionCols + believeCols

	floatCols = ['employees_ft', 'employees_pt', 'volunteers_num_annual', 'constituents']
	floatCols += satisfactionCols
	for col in floatCols:
		projects_df[col] = projects_df[col].astype(float)

	# grant categories
	# - brand, strategy, brochure, annual report, website, finance, HR, donor database?
	grantMap = {}
	grantMap['Web 2'] = 'Advanced Website'
	grantMap['Advanced Website'] = 'Advanced Website'
	grantMap['Web'] = 'Basic Website'
	grantMap['Basic Website'] = 'Basic Website'
	grantMap['Annual Report'] = 'Annual Report'
	grantMap['Brochure'] = 'Brochure'
	grantMap['Brochure/Pamphlet'] = 'Brochure'
	grantMap['Press Kit'] = 'Brochure'
	grantMap['Strategic Planning Prep'] = 'Strategy'
	grantMap['Strategic Scorecard'] = 'Strategy'
	grantMap['Strategic Staff Development'] = 'Strategy'
	grantMap['Competitor/Collaborator Analysis'] = 'Strategy'
	grantMap['Human Resources (HR) Capacity Build'] = 'HR'
	grantMap['Board Recruitment'] = 'HR'
	grantMap['Performance Management'] = 'HR'
	grantMap['Visual Identity & Brand Strategy'] = 'Naming, Visual Identify, and Brand'
	grantMap['Naming & Visual Identity'] = 'Naming, Visual Identify, and Brand'
	grantMap['Key Messages & Brand Strategy'] = 'Naming, Visual Identify, and Brand'
	grantMap['Naming'] = 'Naming, Visual Identify, and Brand'
	grantMap['Financial Analysis'] = 'Financial Analysis'
	grantMap['Donor Database'] = 'Donor Database'

	projects_df['requested_grant_type_normalized'] = projects_df['requested_grant_type'].apply(lambda x: grantMap[x])
	projects_df['assigned_grant_type_normalized'] = projects_df['assigned_grant_type'].apply(lambda x: grantMap[x])

	# building employee features
	projects_df['employees_ft_log'] = transformFeature(projects_df['employees_ft'], 'log')
	projects_df['employees_pt_log'] = transformFeature(projects_df['employees_pt'], 'log')
	projects_df['employees_total'] = projects_df['employees_ft'] + projects_df['employees_pt']
	projects_df['employees_total_log'] = transformFeature(projects_df['employees_total'], 'log')
	projects_df['employees_pt2ft_ratio'] = projects_df['employees_pt'] / projects_df['employees_ft']
	projects_df['employees_more_ft'] = None
	projects_df.ix[projects_df['employees_ft'] > projects_df['employees_pt'], 'employees_more_ft'] = 1
	projects_df.ix[projects_df['employees_ft'] <= projects_df['employees_pt'], 'employees_more_ft'] = 0
	projects_df['volunteers_num_annual_log'] = transformFeature(projects_df['volunteers_num_annual'], 'log')
	projects_df['constituents_log'] = transformFeature(projects_df['constituents'], 'log')
	projects_df['constituents2employee_ratio'] = projects_df['constituents'] / projects_df['employees_total']
	projects_df['constituents2employee_ratio_log'] = transformFeature(projects_df['constituents2employee_ratio'], 'log')

	# building financial features
	projects_df['prev_inc_percentEarned'] = projects_df.apply(lambda x: x['prev_inc_earned'] / x['prev_inc_total'] if x['prev_inc_total'] else None, axis=1)
	projects_df['inc_percentEarned'] = projects_df.apply(lambda x: x['inc_earned'] / x['inc_total'] if x['prev_inc_total'] else None, axis=1)
	projects_df['prev_exp_percentProgram'] = projects_df.apply(lambda x: x['prev_exp_program'] / x['prev_exp_total'] if x['prev_exp_total'] else None, axis=1)
	projects_df['exp_percentProgram'] = projects_df.apply(lambda x: x['exp_program'] / x['exp_total'] if x['exp_total'] else None, axis=1)
	projects_df['budget_prev_net'] = projects_df.apply(lambda x: x['prev_inc_total'] - x['prev_exp_total'] if abs(x['prev_inc_total'] - x['prev_exp_total']) < 1000000 else None, axis = 1)
	projects_df['budget_curr_net'] = projects_df.apply(lambda x: x['inc_total'] - x['exp_total'] if abs(x['inc_total'] - x['exp_total']) < 1000000 else None, axis = 1)
	projects_df['budget_pastYearGrowth_inc'] = projects_df.apply(lambda x: x['inc_total'] / x['prev_inc_total'] if x['prev_inc_total'] else None, axis=1)
	projects_df['budget_pastYearGrowth_exp'] = projects_df.apply(lambda x: x['exp_total'] / x['prev_exp_total'] if x['prev_exp_total'] else None, axis=1)
	projects_df['budget_per_employee_ft'] = projects_df.apply(lambda x: x['inc_total'] / x['employees_ft'] if x['employees_ft'] else None, axis=1)
	projects_df['budget_per_employee_total'] = projects_df.apply(lambda x: x['inc_total'] / x['employees_total'] if x['employees_total'] else None, axis=1)
	projects_df['budget_pastYearGrowth_inc_log'] = transformFeature(projects_df['budget_pastYearGrowth_inc'], 'log')
	projects_df['budget_pastYearGrowth_exp_log'] = transformFeature(projects_df['budget_pastYearGrowth_exp'], 'log')
	projects_df['budget_per_employee_ft_log'] = transformFeature(projects_df['budget_per_employee_ft'], 'log')
	projects_df['budget_per_employee_total_log'] = transformFeature(projects_df['budget_per_employee_total'], 'log')

	categoricalCols = ['assigned_grant_type_normalized', 'app assessment', 'post interview assessment', 'issue area']
	categoricalCols += ['have_strategic_plan']
	binaryCols = ['employees_more_ft', 'Executive Sponsor', 'Day-to-day project lead', 'Board representative', "Content lead (such as finance, technical, HR, or copy)", "Other contributing stakeholder (such as functional manager, key stakeholder, or approval team)"]
	continuousCols = ['weeks_to_staffed', 'employees_ft_log', 'employees_pt_log', 'employees_total_log', 'employees_pt2ft_ratio']
	continuousCols += ['volunteers_num_annual_log', 'constituents_log', 'constituents2employee_ratio_log']
	continuousCols += ['prev_inc_percentEarned','inc_percentEarned','prev_exp_percentProgram','exp_percentProgram','budget_prev_net','budget_curr_net']
	continuousCols += ['budget_pastYearGrowth_inc_log','budget_pastYearGrowth_exp_log','budget_per_employee_ft_log','budget_per_employee_total_log']

	# plotting
	if plotFlag:
		# plotCategoricalFeatures(categoricalCols, satisfactionCols)
		# plotCategoricalFeatures(binaryCols, satisfactionCols)
		continuousCols = ['volunteers_num_annual_log', 'employees_total_log', 'employees_ft_log']
		plotContinuousFeatures(continuousCols, satisfactionCols)

	# BUILDING FEATURE DATAFRAME
	features_df = projects_df.ix[:, continuousCols]
	# convert categoricalFeatures to ordered integers
	evalMetric = "How satisfied with the overall experience? "
	for col in categoricalCols:
		print '-'*10
		print col
		g = projects_df.groupby(col)[evalMetric]
		d = {}
		for key, val in zip(g.mean().index, g.mean()):
			d[key] = val
		d_sorted = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
		d_map = {x[1][0]:x[0] for x in enumerate(d_sorted)}
		features_df[col] = projects_df[col].apply(lambda x: d_map[x] if pd.notnull(x) else None)
		for k, v in d_map.items():
			print k, v
		print '-'*10
	# add evalMetric
	features_df['outcome'] = projects_df[evalMetric]
	features_df.ix[features_df['outcome'] < 6, 'outcome'] = 0
	features_df.ix[features_df['outcome'] == 6, 'outcome'] = 1
	############

	# unimportant features
	# 'budget_pastYearGrowth_inc_log','budget_pastYearGrowth_exp_log','budget_per_employee_ft_log','budget_per_employee_total_log'

	algorithmType = 'gbm'

	# split into train/CV sets
	trainingSetSize = .80 # random
	perm = np.random.permutation(features_df.index.values)
	trainingSetLength = np.round(trainingSetSize * features_df.shape[0])
	training_df = features_df.ix[perm[:trainingSetLength], :]
	cv_df = features_df.ix[perm[trainingSetLength:], :]

	# fill missing values
	replaceNA(training_df, cv_df, ['outcome'])

	# train model
	if algorithmType == 'gbm':
		d = {}
		d["verbose"] = 1 
		d["learning_rate"] = 0.01
		d["min_samples_leaf"] = 1 
		d["n_estimators"] = 100
		d["subsample"] = 0.75
		d["max_features"] = "auto" 
		d["max_depth"] = 6
		hyperparameterDict = d
	elif algorithmType == 'lr':
		d = {}
		d["penalty"] = "l2"
		d["C"] = 0.01
		hyperparameterDict = d		

	featureCols = training_df.columns[:-1]

	featureCols = ['prev_inc_percentEarned','employees_ft_log','budget_per_employee_total_log']
	featureCols += ['budget_curr_net','assigned_grant_type_normalized','constituents_log','exp_percentProgram','budget_pastYearGrowth_inc_log']

	xTrain = training_df.ix[:, featureCols].astype(float)
	yTrain = training_df.ix[:, 'outcome'].astype(int)
	if algorithmType == 'gbm':
		gbm = GradientBoostingClassifier(**hyperparameterDict)
		model = gbm.fit(xTrain, yTrain)
	elif algorithmType == 'lr':
		lr = LogisticRegression(**hyperparameterDict)
		model = lr.fit(xTrain, yTrain)
	trainingOutput = model.predict_proba(xTrain)
	trainingScore = roc_auc_score(yTrain, trainingOutput[:,1])

	# test CV
	xCV = cv_df.ix[:, featureCols].astype(float)
	yCV = cv_df.ix[:, 'outcome'].astype(int)
	cvOutput = model.predict_proba(xCV)
	cvScore = roc_auc_score(yCV, cvOutput[:,1])
	print 'Training Score (AUC): %.4f' % trainingScore
	print 'Cross Validation Score (AUC): %.4f' % cvScore

	# most important features
	keys = featureCols
	if algorithmType in ['rf','gbm','extra']:
		vals = np.round(model.feature_importances_/np.max(model.feature_importances_), 3)
	elif algorithmType in ['lr']:
		vals = np.round(np.abs(model.coef_) / np.max(np.abs(model.coef_)) , 3)[0]
	mostImportantFeatures = {key:val for key, val in zip(keys, vals)}
	mostImportantFeatures = sorted(mostImportantFeatures.iteritems(), key=operator.itemgetter(1))
	print '*'*10
	print 'Relative feature importances'
	for (k, v) in mostImportantFeatures:
		print "%0.0f: %s" % (100*v, k)	

	#plotting partial dependence
	if algorithmType == 'gbm':
		nTopFeatures = 10
		topFeatures = [label for label, _ in mostImportantFeatures[-nTopFeatures:]]
		listofkeys = list(keys)
		featureIndex = [listofkeys.index(topFeature) for topFeature in topFeatures]
		fig, axs = plot_partial_dependence(model, xTrain, features=featureIndex, feature_names=xTrain.columns, n_cols=5)
		fig.show()

	# project applications
	'assigned_grant_type'
	'date_applied'
	'date_app_completed_x'
	'app_assessment'
	'post_interview_assessment'
	'project_pitch_x' # unstructured
	'location_name_x'
	'employees_ft'
	'employees_pt'
	'volunteers_num_annual'
	'constituents'
	'issue_area_x'
	'requested_grant_type'
	'grant_need' # unstructured
	'plan_strategy' # unstructured
	'have_strategic_plan'
	'plan_measure_impact' # unstructured
	'plan_alternate' # unstructured
	'resources'  # unstructured
	'prev_inc_earned'
	'prev_inc_individual'
	'prev_inc_corp'
	'prev_inc_gov'
	'prev_inc_other'
	'prev_inc_total'
	'prev_exp_program'
	'prev_exp_fundraising'
	'prev_exp_admin'
	'prev_exp_total'
	'inc_earned'
	'inc_individual'
	'inc_corp'
	'inc_gov'
	'inc_other'
	'inc_total'
	'exp_program'
	'exp_fundraising'
	'exp_admin'
	'exp_total'

	# Awarded information
	'grant_type_x'
	'region_name'
	'date_project_completed'
	'project_rating' # 315 non-null
	'weeks_to_staffed' # 375 non-null
	'weeks_to_closed' # 374 non-null
	'weeks_ahead' # 392 non-null

	# Projects Application - Project Evaluation
	'issue area'
	'app assessment'
	'post interview assessment'
	'requested grant type'

	# Projects Evaluation
	'StartDate'
	'Project Type'
	'Your Name' # is the name [M/F]
	'Your position within the organization (choose closest):', 'Other (please specify)'
	'Executive Sponsor'
	'Day-to-day project lead'
	'Board representative'
	"Content lead (such as finance, technical, HR, or copy)"
	"Other contributing stakeholder (such as functional manager, key stakeholder, or approval team)"
	'Other (please specify).1'
	"How satisfied with the overall experience? "
	"How satisfied with the final deliverable? "
	"How satisfied with the Taproot Team (e.g. great to work with, professional, capable)"
	"How satisfied with the support provided by the Taproot Foundation during the project (e.g. attention from staff, blueprint materials) "
	"How satisfied with how smoothly and on-time the project ran? "
	"How satisfied with how long the overall project took? "
	"How satisfied with the amount of time the project required of you? "
	"I believe that the Service Grant deliverables has/will strengthen our organization's infrastructure or resources"
	"I believe that the Service Grant process has provided or will provide significant learning to me or our management or staff"
	"I believe that the Service Grant process has or will positively change management or staff's attitudes or beliefs (or the organization's culture)"
	"I believe that the Service Grant has led or will lead to positive changes in the activities undertaken by the organization or key people"
	"I believe that taken together the Service Grant will have a significantly positive impact on our organization's management or governance capability (includes management, fundraising, communications, leadership, human resource development, community outreach, etc.)"
	"$50,000 <i>exclusively for capacity building</i> (technical assistance) or a Taproot Foundation Service Grant "
	"$40,000 <i>exclusively for capacity building</i> (technical assistance) or a Taproot Foundation Service Grant "
	"$30,000 <i>exclusively for capacity building</i> (technical assistance) or a Taproot Foundation Service Grant"
	"$20,000 <i>exclusively for capacity building</i> (technical assistance) or a Taproot Foundation Service Grant"
	"$10,000 <i>exclusively for capacity building</i> (technical assistance) or a Taproot Foundation Service Grant"
	"$5,000 <i>exclusively for capacity building</i> (technical assistance) or a Taproot Foundation Service Grant"
	"$2,500 <i>exclusively for capacity building</i> (technical assistance) or a Taproot Foundation Service Grant"
	"$1,000 <i>exclusively for capacity building</i> (technical assistance) or a Taproot Foundation Service Grant"
	"Given your overall experience, would you recommend that a colleague apply for a Taproot Foundation Service Grant?"
	"Please explain why or why not"
	'outcome'
	
	# Projct evaluation features

	endTime = time.time()
	print "%.1f seconds to parse data" % (endTime - startTime)
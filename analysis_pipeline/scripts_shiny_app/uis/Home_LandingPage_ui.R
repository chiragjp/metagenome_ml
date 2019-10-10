fluidPage(
  titlePanel("Comparing a suite of machine learning tools on reference-based and de novo assembled metagenomic microbiome data leads to a variety of classification conclusions"),
  jumbotron_size("Introduction", "<p> This shiny app presents the results of our paper \"Comparing a suite of machine learning tools on reference-based and de novo assembled metagenomic microbiome data leads to a variety of classification conclusions\". We used microbiome information collected at the taxa and the gene level and leveraged several machine learning algorithms to predict several phenotypes in 1,181 samples collected from 330 infants. <p>
<p> The results of the best performing model for each predicted phenotype are summarized in the table below. <p>
<p> The detailed results are organized in four tabs: \"\", \" Prediction performances\", \"Best predictors\", \"Associations between variables\" and \"Fold tuning\". <p>
<p> For the details about each of those indicators, please see the tabs on the navigation bar, and find advice on how to best use the features of the app in the respective windows below. <p>
                 ", size_header=2, button = FALSE),
  hr(),
  
  dataTableOutput("table_home"),
  htmlOutput("text_home"),
  
  fluidRow(
    jumbotron_size("Prediction performances", 
              "<p><b>Hypothesis 1:</b> The microbiome can be used to predict human phenotypes. <p>
               <p><b>Results 1:</b> Age, Delivery Type, Sex and Country of Origin were significantly better predicted when incorporating microbiome data into the models. Antibiotics usage and breastfed status saw so significant improvement in their predictability. <p>
               <p><b>Hypothesis 2:</b> Information at the gene level allows for a better prediction of the phenotypes than information at the taxa level. <p>
               <p><b>Results 2:</b> All phenotypes were significantly better predicted using information at the gene level rather than at the taxa level. <p>
               <p><b>Hypothesis 3:</b> The association between the microbiome predictors and the predicted phenotype can be non-linear. <p>
               <p><b>Results 3:</b> Age was significantly better predicted using non-linear models such as random forests and gradient boosted machines than using a linear model (elastic model). <p> "
                   , size_header=2, buttonID = "performances", buttonLabel = "Details"),
    jumbotron_size("Best predictors",
              "<p><b>Hypothesis:</b> Each phenotype's prediction relies on key biomarkers. <p>
               <p><b>Results:</b> The best predictors for each phenotype were identified. The majority of the predictors selected were at the genetic level rather than at the taxa level. <p>"
                   , size_header=2, buttonID = "predictors", buttonLabel = "Details"),
    jumbotron_size("Associations between variables", 
              "<p><b>Hypothesis:</b> Significant pairwise relationships can be detected between the phenotypes and/or the biomarkers. <p>
               <p><b>Results:</b> Most of the best predictors selected show a pairwise association with the phenotype they helped predict. <p>"
                   , size_header=2, buttonID = "associations", buttonLabel = "Details"), 
    jumbotron_size("Folds tuning", 
              "<p><b>Hypothesis:</b> The outer cross-validation of the nested cross-validation can give significantly different results on each fold. <p>
               <p><b>Results:</b> Although the different folds tend to select similar hyperparameters values and to give similar results, it also happens that a singularity is observed. For example when predicting age using a linear model (Elastic Net (Caret) and Elastic Net 2), the prediction on the third fold gave a highly negative R-Squared value. This anomaly was not associated with atypical hyperparameter values."
                   , size_header=2, buttonID = "folds", buttonLabel = "Details")
  ),
  
  bsModal("Prediction_performances", "Prediction performances", "performances", size = "large",
          HTML("<p> The results relative to the prediction performances can be found by clicking the \"Prediction performances\" tab in the navigation bar. This window provides a description of the figure and the table, as well as explanations concerning the options that can be selected. <p>
                <h1>Figure and Table</h1>
                <p> The figure and the table compare the prediction performances obtained for a specific phenotype, using different predictors and algorithms.<p>
                <h5>Figure</h5>
                <p> The figure is a bar plot that summarizes the results of the table and allows if quick comparison of the prediction performances between the different predictors and algorithms. <p>
                <h5>Table</h5>
                <p>The table provides the exact numeric values and confidence intervals for each prediction using the selected metric.<p>
                <h1>Options</h1>
                <h5>Select the predicted phenotype</h5>
                <p>Display the results for the prediction of the selected phenotype, to be chosen between Age, Antibiotics Usage, Exclusive breasfeeding, Delivery type, Sex and Country of Origin.<p>
                <h5>Select training or display</h5>
                <p>Display the predictions performances on the training set, or on the testing set.<p>
                <h5>Display the performances with or without confidence intervals</h5>
                <p>Confidence intervals were computed by bootstrapping. Display these confidence invervals along the performance metrics, or not.<p>
                <h5>Select the metric</h5>
                <p>Select the metric used to evaluate the performance of the predictions. <p>
                <p> For regression tasks (Age), only R-Squared is available. <p>
                <p> For binomial classification tasks (Antibiotics Usage, Exclusively Breastfed, Delivery Type and Sex), the available metrics are the AUC under the ROC, the mean class accuracy (the mean of the accuracy on each category), the accuracy, the sensitivity, the specificity, and the cross-entropy. <p>
                <p> For multinomial classification tasks (Country of Origin), the available metrics are the mean class accuracy (the mean of the accuracy on each category), the accuracy, the accuracy on each category, and the cross entropy. <p>
                <h5>Group barplots by:</h5>
                <p>On the barplot figure, group the performances by predictors, or by algorithms.<p>
               ")
  ),
  
  bsModal("Best_predictors", "Best predictors", "predictors", size = "large",
          HTML("<p> The results relative to the best predictors can be found by clicking the \"Best predictors\" tab in the navigation bar. This window provides a description of the table(s) displayed, as well as explanations concerning the options that can be selected. <p>
                <h1>Tables</h1>
                <h5>Table 1</h5>
                <p> Table 1 is only dislayed if the predictors selected are \"Mixed Predictors\" or \"Mixed Predictors + Demographics\". Mixed predictors models are models for which a mix of predictors of different categories (demographic variables, taxa, genes, CAGs and pathways) were available for the model to predict the phenotype of interest. This first table displays the distribution of the best N predictors between the different predictor categories. The first row displays the number of best predictors in each category. The second row displays the percentage of best predictor in each category. Finally, the last row displays the percentage of best predictor in each category, weighted by the absolute value of the regression coefficients (elastic nets) or by the relative variables importances (random forests or gradient boosted machines models). This last row is arguably the best way to analyze which category of predictors holds the greatest predictive power.<p>
                <h5>Table 2</h5>
                <p> This tables ranks the predictors based on the metric selected (usually the absolute value of the regression coefficients for elastic nets, or the relative importance of the variables for random forests and gradient boosted machines).<p>
                <h1>Options</h1>
                <h5>Select the predicted phenotype</h5>
                <p>Display the predictors for the prediction of the selected phenotype, to be chosen between Age, Antibiotics Usage, Exclusive breasfeeding, Delivery type, Sex and Country of Origin.<p>
                <h5>Select the predictors</h5>
                <p> Choose which set of predictors were used to predict the phenotype. The \"Mixed Predictors\" refer to models that incorporated both demographics variables, taxa, genes, cags and pathways. <p>
                <h5>Select the algorithm used for prediction</h5>
                <p> Choose the algorithm that was used to predict the phenotype. The only algorithms avaiable for selection are the elastic nets, the random forests and the gradient boosted machines, because only these model allow a ranking of the predictors they leveraged to predict the target. Support vector machines, K-nearest neighbors and Naive Bayes offer no such option. <p>
                <h5>Select the cross validation fold:</h5>
                <p>It is possible to see which fold was used to test the model. The remaining folds were therefore used to train the model and select the best predictors. For example by selection the fold 1, one displays the best predictors selected by the model trained on the folds 2-10. By default, the tables displays the best predictors selected on the model that was trained using all the samples, so every fold from 1 to 10. <p>
                <h5>Select the metrics to be displayed:</h5>
                <p>This option is only available if more than one metric is available to rank the predictors. This is the case for some phenotypes such as multinomial classifications (Country of Origin). Several metrics are available such as the mean decrease accuracy, the mean decrease Gini indicator, or the decrease in accuracy on a specific category. This option can be used to select which of these metrics will be displayed on the table. <p>
                <h5>Select the metric to order the predictors:</h5>
                <p> This option is only available if more than one metric is available to rank the predictors. Select which of the metric will be used to rank the predictors. <p>
                <h5>Order by absolute values</h5>
                <p> This option is only avaiable when the algorithm selected was an elastic net. In that case, it is possible to use either the absolute value (default) or the signed value of the regression coefficients to rank the best predictors.<p>
                <h5>Select N to classify the top N predictors by type </h5>
                <p> This option is only available if the predictors selected were mixed (\"Mixed Predictors\" or \"Mixed Predictors + Demographics\"). This option allows to select the number of parameters that should be selected when considering the distribution of the best predictors over the different predictors categories in the first table. For example if N is selected to be equal to 100 (default), the table 1 will display which of the top 100 predictors are demographic variables, taxa, genes, CAGs or pathways. <p>
               ")
  ),
  
  bsModal("Associations", "Associations between variables", "associations", size = "large",
          HTML("<p> The results relative to the pairwise associations between variables can be found by clicking the \"Associations between variables\" tab in the navigation bar. This window provides a description of the figure displayed, as well as explanations concerning the options that can be selected. <p>
                <h1>Figure</h1>
                <p> The figure displays the associations between the two selected variables and computes a p-value associated with the H0 hypothesis: the two selected variables are independent.<p>
                <p> If both variables are quantitative (e.g Age), the figure is a linear regression, and the p-value is the one associated with the slope. <p>
                <p> If one variable is quantitative (e.g Age), and the other is binary (e.g Sexe), the figure is a box plot for each category, and the p-value is computed using a t-test for a difference between the means of the quantitative variable in the two categories of the binary variable. <p>
                <p> If one variable is quantitative (e.g Age), and the other is categorical with more than two categories (e.g Country of Origin), the figure is a box plot for each category, and the p-value is computed using an ANOVA tets for a difference between the means of the quantitative variable in each category of the categorical variable. <p>
                <p> If both variables are categorical (e.g Sexe and Country of Origin), the figure is a barplot, and the p-value is computed using a Chi-Squared test. <p>
                <h1>Options</h1>
                <h5>Predictors</h5>
                <p> Choose which variables will be included in the list of variables below (\"Select the 1st variable\" and \"Select the 2nd variable\"). <p>
                <h5>Phenotype</h5>
                <p> This option is only available if \"Genes\" was selected as Predictors above. The genes predictors were generated from the top CAGs predictors, so the list is different depending on which phenotype these CAGs were predicting, and using which algorithm (see below). <p>
                <h5>Algorithm</h5>
                <p> This option is only available if \"Genes\" was selected as Predictors above. The genes predictors were generated from the top CAGs predictors, so the list is different depending on which phenotype these CAGs were predicting (see above), and using which algorithm. The only algorithms that can be selected are the ones that allow a ranking of the predictors, that is the elastic nets, the random forests and the gradient boosted machines. The support vector machines, the K-nearest neighbors and the naive Bayes algorithms do not offer such a feature.<p>
                <h5>Select the 1st variable</h5>
                <p>Select the first variable that will be plotted.<p>
                <h5>Select the 2nd variable</h5>
                <p> Select the second variable that will be plotted. <p>
               ")
  ),
  
  bsModal("Folds_tuning", "Folds Tuning", "folds", size = "large",
          HTML("<p> The results relative to the tuning of the models on each fold can be found by clicking the \"Folds Tuning\" tab in the navigation bar. This window provides a description of the table displayed, as well as explanations concerning the options that can be selected. <p>
                <h1>Table</h1>
                <p>For a selected model the table displays, on each outer cross-validation fold, the prediction performances, the sample sizes, and the selected hyperparameter values. <p>
                <h1>Options</h1>
                <h5>Select the predicted phenotype:</h5>
                <p>Display the information associated with the tuning of the model predicting the selected phenotype, to be chosen between Age, Antibiotics Usage, Exclusive breasfeeding, Delivery type, Sex and Country of Origin.<p>
                <h5>Select the predictors:</h5>
                <p>Display the information associated with the tuning of the model using the selected predictors.<p>
                <h5>Select the algorithm used for prediction:</h5>
                <p>Display the information associated with the tuning of the model using the selected algorithm.<p>
                <h5>Display the results for the datasets:</h5>
                <p>Display the information about the training and/or the testing set.<p>
                <h5>Display the performances</h5>
                <p>Display the columns about the performance of the model.<p>
                <h5>Include the confidence intervals for the performances</h5>
                <p>This option is only available if the \"Display the performances\" box above was ticked. Display the columns about the confidence intervals of the performances of the model.<p>
                <h5>Display the sample sizes</h5>
                <p>Display the columns about the sample sizes of the model. This can include the sample size on the training and/or the testing size. If the phenotype predicted is categorical, the table can also display the sample sizes in the different categories.<p>
                <h5>Display the hyperparameters values</h5>
                <p>Display the hyperparameters selected during the tuning of the model.<p>
                <h5>Display the following performance metrics:</h5>
                <p> This option is only available if the \"Display the performances\" box above was ticked, and if several performance metrics are available for the selected model. Choose which metrics are included in the table. <p>
               ")
  )
  
)

  
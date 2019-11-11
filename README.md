# Project Title: Movie Revenue Analysis and Prediction with Decision Trees and Regression
## Team Members: Thomas Brownlow, Talha Ahsan, Giuseppe Pantalone, Nabiha Ahsan

---

# 1. Overview of Project
Films play a large role in generating profit and creating new Intellectual properties for media companies however, a multitude of factors including casting, ratings, release date and genre can play a role in a film’s financial success. Anita Elberse explored the role that stars have in the financial success of a film while Neil Terry and co. discussed the role that a variety of factors such as production budget and critical acclaim played as well.[1][2] Despite having all of this information, the large number of factors involved means it can be difficult for producers to predict how successful a film would be. To explore possible solutions to this problem, we decided to see if we could use machine learning algorithms to predict the success of a film to help come up with better earnings forecasts for films. This could prove invaluable in helping production companies decide investment and marketing for films that they produce. Our plan is as follows:
1. Use decision trees/random forests for feature selection to find the features that impact revenue the most
2. Use regression to predict the revenue a movie will produce given its features.

---
# 2. Exploration and Visualization of Data
### A) Data: Movie Industry, Three Decades of Movies (Kaggle dataset listed in refrences at bottom)
#### Features of the data, 15 total
1) Budget: Amount spent to produce the movie
2) Company: Production company for the movie
3) Country: Country the movie was produced in
4) Director: Director of movie
5) Genre: Genre of the movie produced
6) Gross: Gross revenue of the movie
7) Name: Name of the movie
8) Rating: Parental Guidance rating of the movie
9) Released: Exact release date of the movie
10) Runtime: Total runtime in minutes of the movie
11) Score: Score (out of 10) on IMDB
12) Star: Leading Actor/Actress for the movie
13) Votes: Number of votes (for the score) on IMDB
14) Writer: Writer of the movie
15) Year: Year the movie was released

### B) Data Visualization
Below are graphs visualizing the distribution each of the features. There are 6820 movies in total, but a decent amount of those have no gross revenue listed which makes them useless for our project. After removing these blank data points we are left with 4638 movies to visualize. You will notice that our features cover of very wide range of values so normalization will be a must. We have continous variables(budget, gross revenue, etc.) and categorical variables(genre, country, company, etc.)

<p float="left">
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/budget.png" width="425" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/company.png" width="425" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/country.png" width="400" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/genre.png" width="400" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/gross.png" width="400" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/length.png" width="400" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/numVotes.png" width="400" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/ratings.png" width="400" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/score.png" width="400" height= "300" />
    <img src="https://github.com/gpantalone/MLTeam28/blob/master/Images/year.png" width="400" height= "300" />
</p>

# 3. Data Pre-Processing
After visualizing the data we were able to process the data in some basic ways before we begin features selection.
1. As noted above we removed all movies with no gross revenue listed
2. Remove repetitve categories (don't need release date when we have release year), and irrelevant categories (movie name).
3. Normalize the data

# 4. Feature Selection


## Movie Profit Analysis

You can use the [editor on GitHub](https://github.com/gpantalone/MLTeam28/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

A machine learning analysis of what factors affect movie profitability. We wanted to see if it was possible to accurately predict the success of a movie release based on available information about the production. We also wanted to see if it was possible to determine the level of impact that certain factors have on success. For example do good stars tend to produce successful movies more than good directors or writers? Are certain combinations of factors more impactful together than they are in isolation?

### Data Visualization

To find a way to appropriately analyze data from the set we started with some histograms to help us look at how much info we felt we could glean from different features.

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/gpantalone/MLTeam28/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

### PCT alternatives

| Quiz Name                          | Link                                                                 | Data                                             |
|-----------------------------------|----------------------------------------------------------------------|--------------------------------------------------|
| GoToQuiz Political Spectrum Quiz  | https://www.gotoquiz.com/politics/political-spectrum-quiz.html      | [data/gotoquiz.json](data/gotoquiz.json)         |
| 10Groups Political Test           | https://10groups.github.io/                                         | [data/10groups.json](data/10groups.json)         |
| SapplyValues                      | https://sapplyvalues.github.io/                                     | [data/sapplyvalues.json](data/sapplyvalues.json) |
| 8Values                           | https://8values.github.io/                                          | [data/8values.json](data/8values.json)           |
| InfiHeal Political Compass Test   | https://www.infiheal.com/personality-test/political-compass/test    | [data/infiheal.json](data/infiheal.json)         |
| The Advocates' World's Smallest Quiz | https://www.theadvocates.org/quiz/                               | [data/advocates.json](data/advocates.json)       |
| YouGov Political Survey           | https://survey.yougov.com/vnkB8FBNQ2hKht#https://isurvey-us.yougov.com/refer/v3mycG7sMW0Ny8 | [data/yougov.json](data/yougov.json)             |
| IDRlabs Political Coordinates Test| https://www.idrlabs.com/political-coordinates/test.php              | [data/idrlabs.json](data/idrlabs.json)           |

This is a blog post with Grok: [Grok Political Preferences (David Rozado)](https://davidrozad.substack.com/p/the-political-preferences-of-grok)

#### GotoQuiz
- https://www.gotoquiz.com/politics/political-spectrum-quiz.html
- Info: There's a likert scale (5) question about "How much does this issue matter?"

#### 10groups, SapplyValues, 8Values
- tried to use selenium, seems not needed at all, they store it in a json file.
- [code](scraping-codes/scrape-10-groups.py). A bunch of websites follow similar pattern, i.e., load the questions from a JS file. See [urls.yaml](scraping-codes/urls.yaml)

#### InfiHeal
- This dataset produces no score, but a position in the PCT scale as an image.

#### Advocates
- To see the results from this dataset, you need to login. I am not sure if that's possible.

#### Yougov
- The yougov data is incomplete because it breaks some of the patterns. If we use it, we would have to figure out how to send this data to yougov.

#### IDRLabs
- This also gives the results in a PCT like scale as InfiHeal, but I am not sure if they are all using the same scale.
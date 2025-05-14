# Assignment

Given the tags and sentences provided, write the code to tag the sentences with the appropriate tags.

To complete the task, copy  the repository code and implement the solution in your own repository.  The code should
be written in python. Any python libraries can be used but if non-standard libraries are used please also
create a *requirements.txt* file with the library name and version and include it with your submission.  

Please include all the code used and provide comments or README as appropriate.


# Data

[tags](data/tags.csv) contains the list of ids, tag names and keywords in csv format. Keywords are array of strings
encoded as a string. For example,

```aiignore
44,Death of a Relative,"['died', 'death', 'funeral', 'payable on death', 'death certificate', 'passed away', 'passed on', 'estate', 'beneficiary', 'probate']"
```

[sentences](data/sentences.csv) contains the list of sentences, one per line.
```aiignore
Loan payoff
how do i dispute a charge?
Is the system down now?
```

# Task 1

Given the data above tag each sentence with all the tags based on exactly matching keywords ignoring the lower/upper case differences.  The output should
be
```aiignore
sentence\ttag1, tag2, tag3
```
i. each sentence followed by a TAB and then comma-delimited list of tags that match.  If no tags apply, the output line
should be just the sentence.

Here is example of the output using couple sentences from [sentences](data/sentences.txt) with comments above each sentence
```aiignore
...
# due to match of 'car loan' in the keywords for  "Vehicle Loan" tag and 'interest rate' in the keywords for "Credit Card"
I need to know my interest rate on my car loan\tVehicle Loan, Credit Card
# no tag assigned since no matches were found for any keywords in tags.csv
Change of address\t
#  due to match of 'CD' in the keywords for "Investing"
CD rates\tInvesting
...
```

The output file should be named: *task_1_output.tsv* 

# Task 2

Use any machine learning based method to tag each sentence with the tags based on that method you chose and output in 
the same format as above:
```aiignore
sentence\ttag1, tag2, tag3
```
ie each sentence followed by a TAB and then comma-delimited list of tags that match.  If no tags apply, the output line
should be just the sentence.

The output file should be named: *task_2_output.tsv* 

If the method requires training, please include both the training and tagging code in your submission. 

# Submission Guidelines
Commit and push to your own repo and send us the repo link. Please do not fork or use our name in your repo.





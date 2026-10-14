// All logic lives in the Jenkins Shared Library (repo: jenkins-shared-library).
// Configure it in Jenkins as a global library named 'mlops-shared-lib'.
@Library('mlops-shared-lib') _

mlopsPipeline(
    datasets:   ['disease', 'boston'],
    recipients: '70155033@student.uol.edu.pk',      // <- change this
    mlflowUri:  'http://127.0.0.1:5000',
    threshold:  '0.85'
)

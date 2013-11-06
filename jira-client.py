import requests
import json

TESLA_PID = '12196'

ISSUE_TYPES = {
	'EPIC': 6,
	'STORY': 7,
	'TASK': 3,
	'BUG': 1
}

USERS = {
	"matt": "selfma",
	"mac": "angemj2",
	"tim": "schwta2",
	"rayland": "ujeanra",
	"saurav": "uvaidsa",
	"kelly": "kelly.jude"
}

class JiraProxy():

	base_url = 'http://jira.pearsoncmg.com/jira/rest/api/2'

	def __init__(self):
		self.creds = self.credentials

	def credentials(self):
		f = open('cred.txt', 'r')
		creds = json.loads(f.read())
		f.close()
		return creds

	def _make_request(self, method, route, payload=None):
		creds = self.credentials();
		auth = (creds['username'], creds['password'])

		url = self.base_url + route
		headers = {'content-type': 'application/json'}

		if method == 'POST':
			data = json.dumps(payload)
			r = requests.post(url, headers=headers, data=data, auth=auth)
		elif method == 'GET':
			r = requests.get(url, headers=headers, auth=auth)
		elif method == 'DELETE':
			r = requests.delete(url, headers=headers, auth=auth)

		if r.status_code < 400:
			return r.json()
		else:
			return r

	def create_issue(self, issue_type, summary, description, assignee):
		payload = {
			"fields": {
				"project": {
					"id": TESLA_PID
				},
				"summary": summary,
				"description": description,
				"issuetype": {
					"id": issue_type
				},
				"assignee": {
					"name": assignee
				}
		   }
		}

		return self._make_request('POST', '/issue/%s' % payload)

	def get_issue(self, issue_id):
		return self._make_request('GET', '/issue/%s' % issue_id)

	def delete_issue(self, issue_id):
		return self._make_request('DELETE', '/issue/' + issue_id)


if __name__ == '__main__':

	summary = "No REST for the Wicked."
	desc = "Creating of an issue using ids for projects and issue types using the REST API"

	jp = JiraProxy()
	# task = jp.create_issue(ISSUE_TYPES['TASK'], summary, desc, "selfma")
	# print task
	# print jp.get_issue(task['id'])

	print jp.delete_issue('345814')
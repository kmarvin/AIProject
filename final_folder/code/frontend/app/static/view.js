'use strict';

/*
 * todo: leere strings anzeigen
 * error anzeigen
 * gleiche string aussortieren
 * */

angular.module('myApp.view', ['ngRoute'])

.config(['$routeProvider', function ($routeProvider) {
    $routeProvider.when('/', {
        templateUrl: '/static/view.html',
        controller: 'ViewCtrl'
    });
}])

.controller('ViewCtrl', function ($scope, $http) {
    $scope.suggestions = [];
    $scope.input = {text: ""};
    $scope.settings = {numberSuggestions: 1};

    $scope.updateInputText = function() {
        var text = $scope.input.text;

        $http.post('/computeInput', {text: text, settings: $scope.settings}).then(function successCallback(response) {
            $scope.input.error = false;
            var data = response.data;
            if(Object.keys(data).length < $scope.settings.numberSuggestions) {
				$scope.input.error = "Through high probability there are less suggestions than requested";
			}
			$scope.suggestions = data.map(function(elem) {
				return {text: elem};
			});
        }, function errorCallback(response) {
            $scope.input.error = "Sorry, we got a Server Error";
        });

        console.log(text);
    }

    $scope.selectText = function(text) {
        $scope.input.text += text + " ";
        $scope.suggestions = [];
        $("#userInput").focus();
    }
});

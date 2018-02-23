'use strict';

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
        $scope.error = false;

        $scope.updateInputText = function () {
            var text = $scope.input.text;

            $http.post('/computeInput', {
                text: text,
                settings: $scope.settings
            }).then(function successCallback(response) {
                $scope.error = false;
                var data = response.data;
                var uniqueData = data.filter(function (item, pos) {
                    return data.indexOf(item) === pos;
                });
                if (uniqueData.length < parseInt($scope.settings.numberSuggestions)) {
                    $scope.error = {
                        message: "Through high probability there are less suggestions than requested",
                        status: "warning"
                    };
                }
                $scope.suggestions = uniqueData.map(function (elem) {
                    var placeholder = elem;
                    if (elem === " ") {
                        placeholder = "_";
                    }
                    return {text: elem, placeholder: placeholder};
                });
            }, function errorCallback() {
                $scope.error = {message: "Sorry, we got a Server Error", status: "error"};
            });
        };

        $scope.selectText = function (text) {
            $scope.input.text += text;
            if (text !== " ") {
                $scope.input.text += " ";
            }
            $scope.suggestions = [];
            $("#userInput").focus();
        }
    });

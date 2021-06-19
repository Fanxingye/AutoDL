@Library('apulis-build@master') _
buildPlugin ( {
    repoName = 'autodl'
    project = ["songshanhu"]
    dockerImages = [        
        [
            'compileContainer': '',
            'preBuild':[],
            'imageName': 'apulistech/autodl',
            'directory': '.',
            'dockerfilePath': 'docker/autogluon.Dockerfile',
            'arch': ['amd64','arm64']
        ]
    ]
})
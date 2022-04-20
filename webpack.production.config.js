const HtmlWebpackPlugin = require('html-webpack-plugin');
const path = require('path');

module.exports = {
    entry: './src/app.js',
    output: {
        filename: 'app.js',
        path: path.resolve(__dirname, './dist'),
        publicPath: '',
        clean:true
    },
    mode: 'production',
    module: {
        rules:[
            {
                test: /\.(png|jpg)$/, //regex to look for png or jpg
                type: 'asset',
                parser: {
                    dataUrlCondition:{
                        maxSize: 3 * 1024 // 3 kilobytes
                    }
                }
            },
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use:{
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/env'],
                        plugins: ['@babel/plugin-proposal-class-properties']
                    }
                }
            }
        ]
    },
    plugins: [
        new HtmlWebpackPlugin({
            title: 'KNN Classification - ML5js'
        })
    ]
}
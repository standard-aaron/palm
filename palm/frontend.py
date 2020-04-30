"""Constructs parsers for the command line interface."""
import logging
from argparse import ArgumentParser
from palm import VERSION
from palm.snp_likelihood import _main as snp_likelihood 
from palm.snp_likelihood import _args as snp_likelihood_args
from palm.palm import _main as palm 
from palm.palm import _args as palm_args 
from palm.snp_extract import _args as snp_extract_args
from palm.snp_extract import _main as snp_extract

COMMANDS = {
    'trait': {'cmd': palm, 'parser': palm_args},
    'coal': {'cmd': snp_extract, 'parser': snp_extract_args},
    'lik': {'cmd': snp_likelihood, 'parser': snp_likelihood_args},
}

def main():
    """
    Takes command line input and calls appropriate clues command.
    The available commands are:
        coal: extract coalescence times from sampled local trees (operates on a single SNP) 
	lik: calculates selection likelihood function (operates on a single SNP) 
        trait: estimate selection on trait(s) (operates on a SET of SNPs)
    """
    parser = ArgumentParser(
        description="""
                    palm v%s is software for estimating selection on SNPs 
                    and/or complex traits.
                    """ % VERSION,
        usage='palm <command> <options>'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    for cmd in COMMANDS:
        cmd_parser = COMMANDS[cmd]['parser'](subparsers)

    args = parser.parse_args()
    try:
        func = COMMANDS[args.command]['cmd']
    except KeyError:
        parser.print_help()
        exit()
    func(args)


if __name__ == '__main__':
    main()
